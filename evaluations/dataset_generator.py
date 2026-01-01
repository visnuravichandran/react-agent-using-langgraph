"""
Test Dataset Generator for Agent Evaluation

This module provides utilities to create and validate test datasets for evaluating
the LangGraph ReAct agent.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


# ============================================================================
# Data Models
# ============================================================================

class TestCase(BaseModel):
    """Schema for a single test case in the evaluation dataset."""

    id: str = Field(..., description="Unique identifier for the test case")
    category: str = Field(..., description="Category: knowledge_base_only, web_search_only, combined_tools, conversational, edge_case")
    query: str = Field(..., description="User query to test")
    expected_output: str = Field(..., description="Expected response content (can be partial match)")
    expected_tools: List[str] = Field(..., description="List of expected tools to be called")
    expected_tool_order: List[str] = Field(..., description="Expected order of tool calls")
    expected_num_steps: int = Field(..., ge=1, description="Expected number of reasoning steps")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")
    reasoning: Optional[str] = Field(None, description="Explanation of why this test case is designed this way")

    @validator("category")
    def validate_category(cls, v):
        valid_categories = {"knowledge_base_only", "web_search_only", "combined_tools", "conversational", "edge_case"}
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v

    @validator("expected_tools", "expected_tool_order")
    def validate_tools(cls, v):
        valid_tools = {"search_knowledge_base", "search_web"}
        for tool in v:
            if tool not in valid_tools and tool != "":  # Allow empty for conversational
                raise ValueError(f"Tool must be one of {valid_tools}")
        return v

    @validator("difficulty")
    def validate_difficulty(cls, v):
        valid_difficulties = {"easy", "medium", "hard"}
        if v not in valid_difficulties:
            raise ValueError(f"Difficulty must be one of {valid_difficulties}")
        return v


class EvaluationDataset(BaseModel):
    """Schema for the complete evaluation dataset."""

    test_cases: List[TestCase] = Field(..., description="List of test cases")
    version: Optional[str] = Field(None, description="Dataset version")

    def save_to_file(self, file_path: str):
        """Save dataset to JSON file."""
        with open(file_path, "w") as f:
            json.dump({"test_cases": [tc.dict() for tc in self.test_cases]}, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(test_cases=[TestCase(**tc) for tc in data["test_cases"]])

    def get_category_distribution(self) -> Dict[str, int]:
        """Get count of test cases by category."""
        distribution = {}
        for tc in self.test_cases:
            distribution[tc.category] = distribution.get(tc.category, 0) + 1
        return distribution

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get count of test cases by difficulty."""
        distribution = {}
        for tc in self.test_cases:
            distribution[tc.difficulty] = distribution.get(tc.difficulty, 0) + 1
        return distribution

    def print_summary(self):
        """Print summary statistics of the dataset."""
        print(f"\n{'='*80}")
        print(f"Dataset Summary: {len(self.test_cases)} test cases")
        print(f"{'='*80}")

        print("\n📊 Category Distribution:")
        for category, count in sorted(self.get_category_distribution().items()):
            percentage = (count / len(self.test_cases)) * 100
            print(f"  {category:25s}: {count:3d} ({percentage:5.1f}%)")

        print("\n📈 Difficulty Distribution:")
        for difficulty, count in sorted(self.get_difficulty_distribution().items()):
            percentage = (count / len(self.test_cases)) * 100
            print(f"  {difficulty:25s}: {count:3d} ({percentage:5.1f}%)")

        print(f"\n{'='*80}\n")


# ============================================================================
# Sample Test Cases
# ============================================================================

SAMPLE_TEST_CASES = [
    # Knowledge Base Only (30%)
    TestCase(
        id="kb_001",
        category="knowledge_base_only",
        query="What are the strategic shifts happening in the footwear and apparel industry?",
        expected_output="Strategic shifts in the footwear and apparel industry include digital transformation, direct-to-consumer models, sustainability initiatives, and supply chain optimization.",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Industry analysis question that should be answered from internal knowledge base."
    ),
    TestCase(
        id="kb_002",
        category="knowledge_base_only",
        query="Tell me about company policies regarding remote work.",
        expected_output="Company policies on remote work include flexible work arrangements, guidelines for remote collaboration, and requirements for maintaining productivity.",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Company policy question requiring internal knowledge base."
    ),
    TestCase(
        id="kb_003",
        category="knowledge_base_only",
        query="What are the key findings from our Q3 2024 market research report?",
        expected_output="Key findings from Q3 2024 market research include market trends, competitive analysis, and strategic recommendations.",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="medium",
        reasoning="Specific historical document retrieval."
    ),

    # Web Search Only (20%)
    TestCase(
        id="web_001",
        category="web_search_only",
        query="What are the latest AI regulation developments announced today?",
        expected_output="Recent AI regulations include new policies from various governments regarding AI safety, ethics, and deployment guidelines.",
        expected_tools=["search_web"],
        expected_tool_order=["search_web"],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Explicitly asks for 'today' information requiring real-time web search."
    ),
    TestCase(
        id="web_002",
        category="web_search_only",
        query="What are the current stock market trends for tech companies?",
        expected_output="Current stock market trends show various movements in tech sector including major companies' performance.",
        expected_tools=["search_web"],
        expected_tool_order=["search_web"],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Current market information requires real-time web search."
    ),
    TestCase(
        id="web_003",
        category="web_search_only",
        query="What breaking news happened in the AI industry this morning?",
        expected_output="Recent breaking news in AI includes new product launches, research breakthroughs, and industry announcements.",
        expected_tools=["search_web"],
        expected_tool_order=["search_web"],
        expected_num_steps=2,
        difficulty="medium",
        reasoning="Breaking news explicitly requires web search."
    ),

    # Combined Tools (30%)
    TestCase(
        id="combined_001",
        category="combined_tools",
        query="How does our company's AI strategy compare to current industry trends?",
        expected_output="Our company's AI strategy focuses on specific areas compared to industry trends showing different approaches and priorities.",
        expected_tools=["search_knowledge_base", "search_web"],
        expected_tool_order=["search_knowledge_base", "search_web"],
        expected_num_steps=3,
        difficulty="medium",
        reasoning="Requires both internal strategy (KB) and external trends (web)."
    ),
    TestCase(
        id="combined_002",
        category="combined_tools",
        query="Compare our product offerings to what competitors are launching today.",
        expected_output="Our products include specific features while competitors are launching new products with different capabilities.",
        expected_tools=["search_knowledge_base", "search_web"],
        expected_tool_order=["search_knowledge_base", "search_web"],
        expected_num_steps=3,
        difficulty="hard",
        reasoning="Requires internal product info and current competitor information."
    ),
    TestCase(
        id="combined_003",
        category="combined_tools",
        query="What are the latest sustainability trends and how do they align with our environmental policies?",
        expected_output="Latest sustainability trends include various environmental initiatives compared to our company's environmental policies and commitments.",
        expected_tools=["search_knowledge_base", "search_web"],
        expected_tool_order=["search_knowledge_base", "search_web"],
        expected_num_steps=3,
        difficulty="medium",
        reasoning="Requires both external trends and internal policies."
    ),

    # Conversational (10%)
    TestCase(
        id="conv_001",
        category="conversational",
        query="Hello! How are you?",
        expected_output="Hello! I'm doing well. How can I help you today?",
        expected_tools=[],
        expected_tool_order=[],
        expected_num_steps=1,
        difficulty="easy",
        reasoning="Simple greeting requiring no tools."
    ),
    TestCase(
        id="conv_002",
        category="conversational",
        query="Thank you, that was very helpful!",
        expected_output="You're welcome! I'm glad I could help.",
        expected_tools=[],
        expected_tool_order=[],
        expected_num_steps=1,
        difficulty="easy",
        reasoning="Conversational response requiring no tools."
    ),
]


# ============================================================================
# Generator Functions
# ============================================================================

def generate_full_dataset() -> EvaluationDataset:
    """
    Generate a complete evaluation dataset with 50+ test cases.

    This function creates a balanced dataset across all categories:
    - 30% Knowledge Base Only
    - 20% Web Search Only
    - 30% Combined Tools
    - 10% Conversational
    - 10% Edge Cases

    Returns:
        EvaluationDataset: Complete dataset with 50+ test cases
    """
    # Start with sample test cases
    test_cases = SAMPLE_TEST_CASES.copy()

    # Add more knowledge base queries
    kb_queries = [
        ("kb_004", "What are the key performance indicators for Q2 2024?", "easy"),
        ("kb_005", "Explain our data privacy and security policies.", "medium"),
        ("kb_006", "What were the main outcomes of the annual strategy meeting?", "medium"),
        ("kb_007", "Describe our customer acquisition strategy.", "medium"),
        ("kb_008", "What are the core values of our organization?", "easy"),
        ("kb_009", "Tell me about our product development process.", "medium"),
        ("kb_010", "What are the training requirements for new employees?", "easy"),
        ("kb_011", "Summarize our financial performance in the last fiscal year.", "hard"),
        ("kb_012", "What are our competitive advantages in the market?", "medium"),
        ("kb_013", "Describe our supply chain management approach.", "hard"),
        ("kb_014", "What are our diversity and inclusion initiatives?", "medium"),
        ("kb_015", "Explain our customer support escalation process.", "medium"),
    ]

    for id, query, diff in kb_queries:
        test_cases.append(TestCase(
            id=id,
            category="knowledge_base_only",
            query=query,
            expected_output="Information from internal knowledge base relevant to the query.",
            expected_tools=["search_knowledge_base"],
            expected_tool_order=["search_knowledge_base"],
            expected_num_steps=2,
            difficulty=diff,
            reasoning="Knowledge base query for internal information."
        ))

    # Add more web search queries
    web_queries = [
        ("web_004", "What are the latest developments in quantum computing?", "medium"),
        ("web_005", "What are today's cryptocurrency market trends?", "easy"),
        ("web_006", "What's the current weather forecast for major cities?", "easy"),
        ("web_007", "What are the breaking news stories in technology today?", "easy"),
        ("web_008", "What are the latest scientific discoveries announced this week?", "medium"),
        ("web_009", "What are current global economic indicators showing?", "medium"),
        ("web_010", "What are the most recent cybersecurity threats reported?", "hard"),
    ]

    for id, query, diff in web_queries:
        test_cases.append(TestCase(
            id=id,
            category="web_search_only",
            query=query,
            expected_output="Current information from web search relevant to the query.",
            expected_tools=["search_web"],
            expected_tool_order=["search_web"],
            expected_num_steps=2,
            difficulty=diff,
            reasoning="Web search query for real-time information."
        ))

    # Add more combined queries
    combined_queries = [
        ("combined_004", "How do our sustainability initiatives compare to industry best practices?", "medium"),
        ("combined_005", "Compare our pricing strategy to current market trends.", "hard"),
        ("combined_006", "How does our R&D investment align with industry standards?", "medium"),
        ("combined_007", "Evaluate our digital transformation progress against industry benchmarks.", "hard"),
        ("combined_008", "How do our employee benefits compare to current market offerings?", "medium"),
        ("combined_009", "Compare our innovation strategy to emerging industry trends.", "hard"),
        ("combined_010", "How does our cybersecurity posture compare to recent threat landscape?", "hard"),
        ("combined_011", "Analyze our market position relative to current competitive dynamics.", "hard"),
        ("combined_012", "How do our customer satisfaction scores compare to industry averages?", "medium"),
    ]

    for id, query, diff in combined_queries:
        test_cases.append(TestCase(
            id=id,
            category="combined_tools",
            query=query,
            expected_output="Combined information from internal knowledge and external web search.",
            expected_tools=["search_knowledge_base", "search_web"],
            expected_tool_order=["search_knowledge_base", "search_web"],
            expected_num_steps=3,
            difficulty=diff,
            reasoning="Comparison query requiring both internal and external information."
        ))

    # Add more conversational queries
    conv_queries = [
        ("conv_003", "What can you help me with?", "easy"),
        ("conv_004", "Thanks!", "easy"),
        ("conv_005", "I appreciate your help.", "easy"),
    ]

    for id, query, diff in conv_queries:
        test_cases.append(TestCase(
            id=id,
            category="conversational",
            query=query,
            expected_output="Conversational response without tool usage.",
            expected_tools=[],
            expected_tool_order=[],
            expected_num_steps=1,
            difficulty=diff,
            reasoning="Conversational interaction requiring no tools."
        ))

    # Add edge cases
    edge_cases = [
        TestCase(
            id="edge_001",
            category="edge_case",
            query="What are your capabilities and can you also tell me about today's news?",
            expected_output="Information about capabilities and current news.",
            expected_tools=["search_web"],
            expected_tool_order=["search_web"],
            expected_num_steps=2,
            difficulty="medium",
            reasoning="Multi-intent query that should prioritize the informational request."
        ),
        TestCase(
            id="edge_002",
            category="edge_case",
            query="",
            expected_output="Could you please provide a question or request?",
            expected_tools=[],
            expected_tool_order=[],
            expected_num_steps=1,
            difficulty="easy",
            reasoning="Empty query edge case."
        ),
        TestCase(
            id="edge_003",
            category="edge_case",
            query="asdfghjkl qwerty",
            expected_output="I'm not sure I understand. Could you please rephrase your question?",
            expected_tools=[],
            expected_tool_order=[],
            expected_num_steps=1,
            difficulty="easy",
            reasoning="Nonsensical query edge case."
        ),
        TestCase(
            id="edge_004",
            category="edge_case",
            query="Tell me about our AI strategy and the latest AI news and how they compare.",
            expected_output="Information about internal AI strategy compared to latest AI news.",
            expected_tools=["search_knowledge_base", "search_web"],
            expected_tool_order=["search_knowledge_base", "search_web"],
            expected_num_steps=3,
            difficulty="hard",
            reasoning="Complex multi-part query requiring careful handling."
        ),
    ]

    test_cases.extend(edge_cases)

    return EvaluationDataset(test_cases=test_cases)


def generate_small_dataset() -> EvaluationDataset:
    """
    Generate a small evaluation dataset with 10 representative test cases.

    Returns:
        EvaluationDataset: Small dataset for quick iteration
    """
    return EvaluationDataset(test_cases=SAMPLE_TEST_CASES[:10])


def validate_dataset(file_path: str) -> bool:
    """
    Validate a dataset file.

    Args:
        file_path: Path to the dataset JSON file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        dataset = EvaluationDataset.load_from_file(file_path)
        print(f"✅ Dataset is valid: {len(dataset.test_cases)} test cases")
        dataset.print_summary()
        return True
    except Exception as e:
        print(f"❌ Dataset validation failed: {str(e)}")
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """Generate and save evaluation datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation datasets for agent testing")
    parser.add_argument("--output-dir", default="datasets", help="Output directory for datasets")
    parser.add_argument("--validate", help="Validate an existing dataset file")

    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.validate)
        return

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate full dataset
    print("Generating full evaluation dataset...")
    full_dataset = generate_full_dataset()
    full_path = os.path.join(args.output_dir, "evaluation_dataset.json")
    full_dataset.save_to_file(full_path)
    print(f"✅ Saved full dataset to: {full_path}")
    full_dataset.print_summary()

    # Generate small dataset
    print("\nGenerating small evaluation dataset...")
    small_dataset = generate_small_dataset()
    small_path = os.path.join(args.output_dir, "evaluation_dataset_small.json")
    small_dataset.save_to_file(small_path)
    print(f"✅ Saved small dataset to: {small_path}")
    small_dataset.print_summary()


if __name__ == "__main__":
    main()