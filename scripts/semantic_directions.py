flower_classes = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]  # "common tulip", "wild rose",

# print(len(flower_classes))

semantic_directions_flowers = {
    # "vibrancy": ("a dull [CLASS] flower", "a vibrant [CLASS] flower"),
    # "petal_density": (
    #     "a sparse-petaled [CLASS] flower",
    #     "a densely-petaled [CLASS] flower",
    # ),
    # "background_blur": (
    #     "a [CLASS] flower with a sharp background",
    #     "a [CLASS] flower with a blurred background",
    # ),
    # "brightness": (
    #     "a dark [CLASS] flower photo",
    #     "a bright [CLASS] flower photo",
    # ),
    # "saturation": (
    #     "a desaturated [CLASS] flower",
    #     "a saturated [CLASS] flower",
    # ),
    # "blooming": (
    #     "a slightly blooming [CLASS] flower",
    #     "a fully blooming [CLASS] flower",
    # ),
    # "closeness": (
    #     "a distant photo of a [CLASS] flower",
    #     "a close-up photo of a [CLASS] flower",
    # ),
    # "color_temp": (
    #     "a [CLASS] flower in a cool tone",
    #     "a [CLASS] flower in a warm tone",
    # ),
    # "texture": (
    #     "a soft-focused [CLASS] flower",
    #     "a sharp-textured [CLASS] flower",
    # ),
    # "center_visibility": (
    #     "a [CLASS] flower with its center hidden",
    #     "a [CLASS] flower with its center clearly visible",
    # ),
    # "style": (
    #     "a realistic photo of a [CLASS] flower",
    #     "an artistic photo of a [CLASS] flower",
    # ),
    # "direction": (
    #     "A [flower class] facing left, photographed from the side",
    #     "A [flower class] rotated to the right, showing its side profile",
    # ),
    # "direction": (
    #     "A red flower",
    #     "A yellow flower",
    # ),
    # "size":(
    #     "A small flower",
    #     "A large flower",
    # ),
    "saturation": ("A flower with pink petals", "A flower with white petals")
}


def flower_split():
    labels = sorted(map(str, range(102)))
    base_labels = []
    val_labels = []

    for i, label in enumerate(labels):
        if i % 6 != 1:
            base_labels.append(label)
        if i % 6 == 1:
            val_labels.append(label)

    # flower_classes[int(val_labels[val_l])]
    return base_labels, val_labels


if __name__ == "__main__":
    flower_split()
