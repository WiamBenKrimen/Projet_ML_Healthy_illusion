import pandas as pd

from src.features import prepare_model_input


def test_prepare_model_input_normalizes_and_engineers_features():
    raw = pd.DataFrame(
        [
            {
                "sugars_100g": 10.0,
                "fat_100g": 5.0,
                "saturated_fat_100g": 2.0,
                "salt_100g": 0.5,
                "fiber_100g": 2.0,
                "proteins_100g": 5.0,
                "energy_kcal_100g": 250.0,
                "additives_count": 1,
                "main_category": "Plant Based Foods And Beverages",
                "country": "France",
                "has_labels": 1,
                "image_saine": 0,
            }
        ]
    )
    result = prepare_model_input(raw)
    assert result.loc[0, "main_category"] == "plant-based-foods-and-beverages"
    assert result.loc[0, "country"] == "france"
    assert result.loc[0, "salt_sugar_interaction"] == 5.0
    assert "nutrition_risk_score" in result.columns
