#include <engine/scheduler.hpp>
#include <engine/memory_graph.hpp>
#include <slab/slab_allocator.hpp>
#include <metric/fisher_rao.hpp>
#include <sheaf/coboundary.hpp>
#include <langevin/sde_stepper.hpp>
#include <persist/sqlite_store.hpp>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Global scheduler pointer for signal handler
static slm::engine::Scheduler* g_scheduler = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_scheduler) {
        g_scheduler->request_stop();
    }
}

int main(int argc, char* argv[]) {
    std::string shm_name = "slmfs_shm";
    std::filesystem::path db_path = ".slmfs/memory.db";
    uint32_t shm_size = 4 * 1024 * 1024;
    uint32_t slab_size = 64 * 1024;
    uint32_t ctrl_size = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.starts_with("--shm-name=")) {
            shm_name = arg.substr(11);
        } else if (arg.starts_with("--db-path=")) {
            db_path = arg.substr(10);
        }
    }

    if (!db_path.parent_path().empty()) {
        std::filesystem::create_directories(db_path.parent_path());
    }

    uint32_t slab_count = (shm_size - ctrl_size) / slab_size;

    std::cout << "SLMFS Engine starting...\n"
              << "  shm_name:   " << shm_name << "\n"
              << "  db_path:    " << db_path << "\n"
              << "  slab_count: " << slab_count << "\n"
              << "  slab_size:  " << slab_size << "\n";

    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        std::cerr << "Failed to create shared memory: " << shm_name << "\n";
        return 1;
    }

    if (ftruncate(shm_fd, shm_size) < 0) {
        std::cerr << "Failed to resize shared memory\n";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        return 1;
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap shared memory\n";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        return 1;
    }

    slm::slab::SlabAllocator slab(shm_ptr, slab_count, slab_size, ctrl_size);
    slm::engine::MemoryGraph graph;
    slm::metric::FisherRaoMetric metric;
    slm::sheaf::CoboundaryOperator sheaf;
    slm::langevin::LangevinStepper langevin({
        .dt = 5.0f,
        .lambda_decay = 0.01f,
        .noise_scale = 0.001f,
        .archive_threshold = 0.95f,
    });
    slm::persist::SqliteStore store(db_path);

    store.load(graph);
    std::cout << "Loaded " << graph.size() << " nodes from " << db_path << "\n";

    slm::engine::Scheduler scheduler(
        slab, slab.cmd_queue(), graph, metric, sheaf, langevin, store, {}
    );

    g_scheduler = &scheduler;
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    std::thread engine_thread([&] {
        scheduler.run();
    });

    std::cout << "Engine running. PID=" << getpid() << "\n";

    engine_thread.join();

    munmap(shm_ptr, shm_size);
    close(shm_fd);
    shm_unlink(shm_name.c_str());

    std::cout << "Engine stopped.\n";
    return 0;
}
