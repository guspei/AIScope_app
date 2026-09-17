"""Leyenda de clases y códigos de metadatos, tomados de la app de etiquetado GDD-app
(github.com/theaiscope/GDD-app: res/values/colors.xml, model/SampleModels.kt)."""

# Color RGB del pincel en la máscara -> estadio. El artefacto no es un parásito.
MASK_COLORS = {
    "#5cbfb0": "ring",
    "#bfbe52": "trophozoite",
    "#bf6b49": "schizont",
    "#946fbf": "gametocyte",
    "#4f6fd0": "artefact",
}
PARASITE_STAGES = ["ring", "trophozoite", "schizont", "gametocyte"]
NOT_PARASITE = ["artefact"]
# Orden de clases del detector (id = índice). El artefacto se entrena pero no suma en el conteo.
CLASSES = PARASITE_STAGES + NOT_PARASITE

# Detector sin estadio: una sola clase y el artefacto pasa a ser fondo.
PARASITE = "parasite"
SINGLE_CLASS = [PARASITE]


def class_map(src: list[str], dst: list[str]) -> dict[int, int]:
    """Ids de `src` → ids de `dst` por nombre. Los estadios caen en «parasite» si `dst` no los tiene; lo que no
    existe en `dst` (p. ej. el artefacto en el detector de una clase) se descarta."""
    out = {}
    for i, name in enumerate(src):
        if name in dst:
            out[i] = dst.index(name)
        elif name in PARASITE_STAGES and PARASITE in dst:
            out[i] = dst.index(PARASITE)
    return out

STAGE_ES = {
    "ring": "anillo",
    "trophozoite": "trofozoíto",
    "schizont": "esquizonte",
    "gametocyte": "gametocito",
    "artefact": "artefacto",
}

# metadata.species (MalariaSpecies.id)
SPECIES = {1: "P. falciparum", 2: "P. vivax", 3: "P. ovale", 4: "P. malariae", 5: "P. knowlesi"}
SPECIES_SHORT = {1: "Pf", 2: "Pv", 3: "Po", 4: "Pm", 5: "Pk"}

# metadata.bloodType (SmearType.id)
SMEAR = {1: "fina", 2: "gruesa"}

# La app de etiquetado pinta con grosor PATH_STROKE_WIDTH / zoom (MaskLayer.kt): un toque deja un disco
# de 80/zoom px en coordenadas de imagen. El tamaño del trazo depende del zoom, no del parásito.
BRUSH_SCREEN_PX = 80
