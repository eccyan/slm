#include <gtest/gtest.h>
#include <engine/memory_graph.hpp>
#include <vector>

using namespace slm::engine;
using namespace slm::langevin;
using namespace slm::metric;

namespace {

struct GraphFixture : public ::testing::Test {
    MemoryGraph graph;

    uint32_t insert_node(const std::string& text, uint32_t parent_id = 0,
                         uint8_t depth = 0, float x = 0.0f, float y = 0.0f) {
        std::vector<float> mu = {1.0f, 0.0f, 0.0f};
        std::vector<float> sigma = {1.0f, 1.0f, 1.0f};
        NodeState state{};
        state.pos = {x, y};
        state.last_access_time = 0.0;
        state.access_count = 0;
        return graph.insert(std::move(mu), std::move(sigma), text,
                            parent_id, depth, state);
    }
};

} // namespace

TEST_F(GraphFixture, InsertReturnsUniqueIds) {
    auto id1 = insert_node("first");
    auto id2 = insert_node("second");
    auto id3 = insert_node("third");
    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id1, id3);
}

TEST_F(GraphFixture, InsertIncrementsSize) {
    EXPECT_EQ(graph.size(), 0u);
    insert_node("a");
    EXPECT_EQ(graph.size(), 1u);
    insert_node("b");
    EXPECT_EQ(graph.size(), 2u);
}

TEST_F(GraphFixture, GetNodeData) {
    auto id = insert_node("hello world", 5, 2, 0.3f, 0.4f);
    EXPECT_EQ(graph.text(id), "hello world");
    EXPECT_EQ(graph.parent_id(id), 5u);
    EXPECT_EQ(graph.depth(id), 2u);
    EXPECT_FLOAT_EQ(graph.state(id).pos.x, 0.3f);
    EXPECT_FLOAT_EQ(graph.state(id).pos.y, 0.4f);
    EXPECT_EQ(graph.mu(id).size(), 3u);
    EXPECT_EQ(graph.sigma(id).size(), 3u);
}

TEST_F(GraphFixture, RemoveNode) {
    auto id1 = insert_node("keep");
    auto id2 = insert_node("remove");
    auto id3 = insert_node("keep too");
    graph.remove(id2);
    EXPECT_EQ(graph.size(), 2u);
    EXPECT_EQ(graph.text(id1), "keep");
    EXPECT_EQ(graph.text(id3), "keep too");
}

TEST_F(GraphFixture, RemoveLastNode) {
    auto id = insert_node("only");
    graph.remove(id);
    EXPECT_EQ(graph.size(), 0u);
}

TEST_F(GraphFixture, SiblingsQuery) {
    auto p = insert_node("parent", 0, 0);
    auto c1 = insert_node("child1", p, 1);
    auto c2 = insert_node("child2", p, 1);
    auto c3 = insert_node("child3", p, 1);
    auto other = insert_node("other", 0, 0);
    auto siblings = graph.siblings(p);
    EXPECT_EQ(siblings.size(), 3u);
    auto has = [&](uint32_t id) {
        return std::find(siblings.begin(), siblings.end(), id) != siblings.end();
    };
    EXPECT_TRUE(has(c1));
    EXPECT_TRUE(has(c2));
    EXPECT_TRUE(has(c3));
    EXPECT_FALSE(has(other));
}

TEST_F(GraphFixture, ParentQuery) {
    auto p = insert_node("parent", 0, 0);
    auto c = insert_node("child", p, 1);
    auto parent = graph.parent(c);
    EXPECT_TRUE(parent.has_value());
    EXPECT_EQ(*parent, p);
}

TEST_F(GraphFixture, ParentOfRootIsNullopt) {
    auto root = insert_node("root", 0, 0);
    auto parent = graph.parent(root);
    EXPECT_FALSE(parent.has_value());
}

TEST_F(GraphFixture, StatesSpanContiguous) {
    insert_node("a", 0, 0, 0.1f, 0.0f);
    insert_node("b", 0, 0, 0.5f, 0.0f);
    insert_node("c", 0, 0, 0.9f, 0.0f);
    auto states = graph.all_states();
    EXPECT_EQ(states.size(), 3u);
    EXPECT_EQ(&states[1] - &states[0], 1);
    EXPECT_EQ(&states[2] - &states[1], 1);
}

TEST_F(GraphFixture, SetAnnotation) {
    auto id = insert_node("test");
    EXPECT_EQ(graph.annotation(id), "");
    graph.set_annotation(id, "<!-- cohomology: ... -->");
    EXPECT_EQ(graph.annotation(id), "<!-- cohomology: ... -->");
}

TEST_F(GraphFixture, AllIds) {
    auto id1 = insert_node("a");
    auto id2 = insert_node("b");
    auto ids = graph.all_ids();
    EXPECT_EQ(ids.size(), 2u);
}
