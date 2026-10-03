"""Tamil Nadu crop names shown alongside, but not conflated with, ML labels."""


TAMIL_NADU_CROPS = [
    {
        "name": "Paddy (rice)",
        "group": "Cereals",
        "model_label": "rice",
        "season": "Kar/Kuruvai (Apr–Jul), Samba/Thaladi (Aug–Nov), Navarai (Dec–Mar)",
        "note": "Choose the season and variety for your district and water source.",
    },
    {
        "name": "Maize",
        "group": "Cereals",
        "model_label": "maize",
        "season": "Season depends on district, rainfall and irrigation",
        "note": "Check the local sowing window and expected water availability.",
    },
    {
        "name": "Sorghum (cholam)",
        "group": "Millets",
        "model_label": None,
        "season": "Often included in rainfed and deficit-rainfall crop plans",
        "note": "Ask the local KVK which duration and variety fits your block.",
    },
    {
        "name": "Pearl millet (cumbu)",
        "group": "Millets",
        "model_label": None,
        "season": "Rainfall and zone dependent",
        "note": "A regional option in several dryland and contingency plans.",
    },
    {
        "name": "Finger millet (ragi)",
        "group": "Millets",
        "model_label": None,
        "season": "Rainfall and zone dependent",
        "note": "Check local sowing dates and short-duration varieties.",
    },
    {
        "name": "Black gram (urad)",
        "group": "Pulses",
        "model_label": "blackgram",
        "season": "Often grown as a pulse or rice-fallow crop",
        "note": "The best window depends on district and preceding crop.",
    },
    {
        "name": "Green gram (moong)",
        "group": "Pulses",
        "model_label": "mungbean",
        "season": "Season depends on district and water availability",
        "note": "Confirm variety and sowing window with the local advisory.",
    },
    {
        "name": "Red gram (pigeon pea)",
        "group": "Pulses",
        "model_label": "pigeonpeas",
        "season": "Rainfed and zone dependent",
        "note": "Check the local crop plan before intercropping or sowing.",
    },
    {
        "name": "Groundnut",
        "group": "Oilseeds",
        "model_label": None,
        "season": "Varies by district and monsoon",
        "note": "The current ML crop model does not classify groundnut.",
    },
    {
        "name": "Gingelly (sesame)",
        "group": "Oilseeds",
        "model_label": None,
        "season": "Varies by district and crop rotation",
        "note": "The current ML crop model does not classify sesame.",
    },
    {
        "name": "Sunflower",
        "group": "Oilseeds",
        "model_label": None,
        "season": "Rainfall and zone dependent",
        "note": "The current ML crop model does not classify sunflower.",
    },
    {
        "name": "Cotton",
        "group": "Commercial",
        "model_label": "cotton",
        "season": "Sowing window varies across agro-climatic zones",
        "note": "Use district-specific sowing and pest advisories.",
    },
    {
        "name": "Sugarcane",
        "group": "Commercial",
        "model_label": None,
        "season": "Planting window varies by zone and water supply",
        "note": "The current ML crop model does not classify sugarcane.",
    },
    {
        "name": "Coconut",
        "group": "Plantation",
        "model_label": "coconut",
        "season": "Perennial crop",
        "note": "Water and nutrient management should match the garden and soil test.",
    },
    {
        "name": "Banana",
        "group": "Horticulture",
        "model_label": "banana",
        "season": "Planting is commonly planned around local water and market windows",
        "note": "Select a variety and planting time for your district.",
    },
    {
        "name": "Chilli",
        "group": "Horticulture",
        "model_label": None,
        "season": "Varies by zone and irrigation",
        "note": "The current ML crop model does not classify chilli.",
    },
    {
        "name": "Tomato",
        "group": "Horticulture",
        "model_label": None,
        "season": "Varies by zone, temperature and irrigation",
        "note": "The current ML crop model does not classify tomato.",
    },
    {
        "name": "Turmeric",
        "group": "Horticulture",
        "model_label": None,
        "season": "Planting window varies by zone",
        "note": "The current ML crop model does not classify turmeric.",
    },
]


def get_tamil_nadu_crop_catalog():
    return [dict(crop) for crop in TAMIL_NADU_CROPS]
