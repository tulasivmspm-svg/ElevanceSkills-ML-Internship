
# =====================================================
# ANIMAL DIET CLASSIFICATION CATEGORIES
# =====================================================

# Carnivores
CARNIVORES = {
    "tiger",
    "lion",
    "leopard",
    "leopard cat",
    "banded linsang",
    "bay cat",
    "mongoose sp",
    "yellow-throated marten",
    "otter civet",
}

# Herbivores
HERBIVORES = {
    "mousedeer",
    "muntjac",
    "sambar deer",
    "african elephant",
    "giraffe",
    "plains zebra",
    "hartebeest",
    "impala",
    "white-fronted langur",
    "long-tailed porcupine",
    "thick-spined porcupine",
}

# Omnivores
OMNIVORES = {
    "human",
    "domestic dog",
    "bearded pig",
    "sun bear",
    "binturong",
    "banded civet",
    "malayan civet",
    "masked palm civet",
    "small-toothed palm civet",
}


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize_name(animal_name):
    return str(animal_name).lower().strip()


def is_carnivore(animal_name):
    return normalize_name(animal_name) in CARNIVORES


def is_herbivore(animal_name):
    return normalize_name(animal_name) in HERBIVORES


def is_omnivore(animal_name):
    return normalize_name(animal_name) in OMNIVORES


def get_diet_category(animal_name):

    name = normalize_name(animal_name)

    if name in CARNIVORES:
        return "carnivore"

    if name in HERBIVORES:
        return "herbivore"

    if name in OMNIVORES:
        return "omnivore"

    return "unknown"
