"""
Test pricing data structure and validity.
This test doesn't require dependencies, just validates the JSON structure.
"""

import json
from pathlib import Path


def test_pricing_data():
    """Test that pricing.json is valid and complete."""
    print("=" * 70)
    print("PRICING DATA VALIDATION TEST")
    print("=" * 70)
    print()

    # Load pricing data
    pricing_file = Path("llmops_monitoring/data/pricing.json")
    print(f"Loading: {pricing_file}")

    with open(pricing_file, 'r') as f:
        data = json.load(f)

    print(f"✓ Valid JSON structure\n")

    # Check structure
    assert "version" in data, "Missing 'version' field"
    assert "last_updated" in data, "Missing 'last_updated' field"
    assert "models" in data, "Missing 'models' field"
    assert "defaults" in data, "Missing 'defaults' field"

    print(f"Version: {data['version']}")
    print(f"Last Updated: {data['last_updated']}")
    print()

    # Check models
    models = data["models"]
    print(f"Total Models: {len(models)}")
    print()

    # Expected models
    expected_models = {
        # OpenAI
        "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
        # Anthropic
        "claude-3-opus", "claude-3-sonnet", "claude-3-5-sonnet", "claude-3-haiku",
        # Google
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro",
        # Meta
        "llama-3-8b", "llama-3-70b",
        # Mistral
        "mixtral-8x7b", "mistral-small", "mistral-medium", "mistral-large"
    }

    assert len(models) == len(expected_models), f"Expected {len(expected_models)} models, got {len(models)}"

    # Check each model
    providers = {}
    print("Model Pricing Summary:")
    print("-" * 70)

    for model_name in sorted(expected_models):
        assert model_name in models, f"Missing model: {model_name}"

        model = models[model_name]

        # Required fields
        assert "provider" in model, f"{model_name}: Missing provider"
        assert "input_cost_per_1k_tokens" in model, f"{model_name}: Missing input cost"
        assert "output_cost_per_1k_tokens" in model, f"{model_name}: Missing output cost"
        assert "context_window" in model, f"{model_name}: Missing context window"
        assert "description" in model, f"{model_name}: Missing description"

        # Track providers
        provider = model["provider"]
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model_name)

        # Display
        input_cost = model["input_cost_per_1k_tokens"]
        output_cost = model["output_cost_per_1k_tokens"]
        context = model["context_window"]

        print(f"{model_name:25s} | {provider:10s} | ${input_cost:8.6f}/1K in | ${output_cost:8.6f}/1K out | {context:>7,} ctx")

    print()

    # Provider summary
    print("Provider Summary:")
    print("-" * 70)
    for provider, model_list in sorted(providers.items()):
        print(f"{provider:10s}: {len(model_list)} models")
    print()

    # Check defaults
    defaults = data["defaults"]
    assert "fallback_cost_per_1k_chars" in defaults, "Missing fallback_cost_per_1k_chars"
    assert "chars_to_tokens_ratio" in defaults, "Missing chars_to_tokens_ratio"
    assert "image_cost_per_image" in defaults, "Missing image_cost_per_image"

    print("Defaults:")
    print(f"  Fallback cost: ${defaults['fallback_cost_per_1k_chars']}/1K chars")
    print(f"  Chars to tokens ratio: {defaults['chars_to_tokens_ratio']}:1")
    print(f"  Image cost: ${defaults['image_cost_per_image']}/image")
    print()

    # Validate pricing values are reasonable
    print("Validating pricing ranges...")
    for model_name, model in models.items():
        input_cost = model["input_cost_per_1k_tokens"]
        output_cost = model["output_cost_per_1k_tokens"]

        # Costs should be positive
        assert input_cost > 0, f"{model_name}: Input cost must be positive"
        assert output_cost > 0, f"{model_name}: Output cost must be positive"

        # Output typically costs more than input
        # (except for some models where they're equal)
        assert output_cost >= input_cost, f"{model_name}: Output cost should be >= input cost"

        # Costs should be reasonable (< $1 per 1K tokens)
        assert input_cost < 1.0, f"{model_name}: Input cost seems too high: ${input_cost}"
        assert output_cost < 1.0, f"{model_name}: Output cost seems too high: ${output_cost}"

        # Context window should be reasonable
        context = model["context_window"]
        assert context >= 1000, f"{model_name}: Context window too small: {context}"
        assert context <= 2000000, f"{model_name}: Context window seems too large: {context}"

    print("✓ All pricing values validated")
    print()

    # Cost comparison
    print("Cost Comparison (Output @ 1M tokens):")
    print("-" * 70)

    costs = []
    for model_name, model in models.items():
        output_cost_per_million = model["output_cost_per_1k_tokens"] * 1000
        costs.append((model_name, output_cost_per_million, model["provider"]))

    for model_name, cost, provider in sorted(costs, key=lambda x: x[1]):
        print(f"${cost:8.2f}  {model_name:25s} ({provider})")

    print()

    print("=" * 70)
    print("✅ ALL PRICING DATA TESTS PASSED!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  ✓ {len(models)} models validated")
    print(f"  ✓ {len(providers)} providers (OpenAI, Anthropic, Google, Meta, Mistral)")
    print(f"  ✓ All pricing values in reasonable ranges")
    print(f"  ✓ All required fields present")
    print(f"  ✓ Defaults configured correctly")
    print()
    print("Pricing data is READY FOR PRODUCTION! 🚀")
    print()


if __name__ == "__main__":
    try:
        test_pricing_data()
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
