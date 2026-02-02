
ORIGINAL_LABEL_NAME_L3 = {
    0: 'Not appeared',
    1: 'Broadleaved Mixed and Yew Woodland',
    2: 'Coniferous Woodland',
    3: 'Boundary and Linear Features',
    4: 'Arable and Horticulture',
    5: 'Improved Grassland',
    6: 'Neutral Grassland',
    7: 'Calcareous Grassland',
    8: 'Acid Grassland',
    9: 'Bracken',
    10: 'Dwarf Shrub Heath',
    11: 'Fen, Marsh, Swamp',
    12: 'Bog',
    13: 'Standing Open Waters and Canals',
    14: 'Not appeared',
    15: 'Montane',
    16: 'Inland Rock',
    17: 'Urban',
    18: 'Supra-littoral Rock',
    19: 'Supra-littoral Sediment',
    20: 'Littoral Rock',
    21: 'Littoral Sediment',
    22: 'Sea'
}

REASSIGN_LABEL_NAME_L3 = {
    0: 'Urban',
    1: 'Broadleaved Mixed and Yew Woodland',
    2: 'Coniferous Woodland',
    3: 'Sea',
    4: 'Arable and Horticulture',
    5: 'Improved Grassland',
    6: 'Neutral Grassland',
    7: 'Calcareous Grassland',
    8: 'Acid Grassland',
    9: 'Bracken',
    10: 'Dwarf Shrub Heath',
    11: 'Fen, Marsh, Swamp',
    12: 'Bog',
    13: 'Littoral Rock',
    14: 'Littoral Sediment',
    15: 'Montane',
    16: 'Standing Open Waters and Canals',
    17: 'Inland Rock',
    18: 'Supra-littoral Rock',
    19: 'Supra-littoral Sediment',
}

ORIGINAL_NAME_LABEL_L3 = {
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Boundary and Linear Features': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Standing Open Waters and Canals': 13,
    'Montane': 15,
    'Inland Rock': 16,
    'Urban': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19,
    'Littoral Rock': 20,
    'Littoral Sediment': 21,
    'Sea': 22
}

REASSIGN_NAME_LABEL_L3 = {
    'Urban': 0,
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Sea': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Littoral Rock': 13,
    'Littoral Sediment': 14,
    'Montane': 15,
    'Standing Open Waters and Canals': 16,
    'Inland Rock': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19
}

NAME_LABEL_L2 = {
    'Urban': 0,
    'Woodland and Forest': 1,
    'Cropland': 2,
    'Grassland': 3,
    'Heathland and Shrub': 4,
    'Wetland': 5,
    'Marine Inlets and Transitional Waters': 6,
    'Sparsely Vegetated Land': 7,
    'Rivers and Lakes': 8,
    'Sea': 9,
    'Montane': 10,
}

REASSIGN_NAME_LABEL_L3L2 = {
    'Urban': (0, 0),
    'Broadleaved Mixed and Yew Woodland': (1, 1),
    'Coniferous Woodland': (2, 1),
    'Sea': (3, 9),
    'Arable and Horticulture': (4, 2),
    'Improved Grassland': (5, 3),
    'Neutral Grassland': (6, 3),
    'Calcareous Grassland': (7, 3),
    'Acid Grassland': (8, 3),
    'Bracken': (9, 3),
    'Dwarf Shrub Heath': (10, 4),
    'Fen, Marsh, Swamp': (11, 5),
    'Bog': (12, 5),
    'Littoral Rock': (13, 6),
    'Littoral Sediment': (14, 6),
    'Montane': (15, 10),
    'Standing Open Waters and Canals': (16, 8),
    'Inland Rock': (17, 7),
    'Supra-littoral Rock': (18, 7),
    'Supra-littoral Sediment': (19, 7),
}

NAME_ABB_L2 = {
    'Urban': 'U',
    'Woodland and forest': 'WLF',
    'Cropland': 'CL',
    'Grassland': 'GL',
    'Heathland and shrub': 'HS',
    'Wetland': 'WL',
    'Marine inlets and transitional waters': 'MITW',
    'Sparsely vegetated land': 'SVL',
    'Rivers and lakes': 'RL',
    'Sea': 'S',
    'Montane': 'M',
}

CORRUPT_IMAGES = [
'ATT3735_594XX3_2023_photo2-20230928-121257.jpg'
]

# Used by the Wandb sweeps config to assign values for the main config
SWEEP_KEY_MAPPING = {
    'cross_valid': ['cross_valid'],
    'first_cv_only': ['data', 'data_split', 'first_cv_only'],
    'num_epochs': ['training', 'num_epochs'],
    'optimiser': ['training', 'optimiser', 'type'],
    'lr': ['training', 'optimiser', 'lr'],
    'weight_decay': ['training', 'optimiser', 'weight_decay'],
    'batch_size': ['data', 'batch_size'],
    'img_resize': ['data', 'preprocessing', 'resize'],
    'model_name': ['model', 'name'],
    'model_config': ['model', 'model_config'],
    'input_size': ['model', 'input_size'],
    'flip': ['data', 'preprocessing', 'augmentations', 'flip'],
    'rotation': ['data', 'preprocessing', 'augmentations', 'rotation'],
    'random_crop': ['data', 'preprocessing', 'augmentations', 'random_crop'],
    'multi_views_supcon': ['data', 'preprocessing', 'multi_views', 'supcon'],
    'supcon_pretrain': ['training', 'supcon_conf', 'pretrain'],
    'supcon_ptr_dir': ['training', 'supcon_conf', 'prt_dir'],
    'supcon_prt_filename': ['training', 'supcon_conf', 'prt_filename'],
}


def l2_names_to_l3(l2_names):
    """
    Convert L2 names (strings) to ordered L3 classnames + L3 ids.
    Uses REASSIGN_NAME_LABEL_L3L2 (L3 -> (L3 id, L2 id)).
    """
    if not l2_names:
        return [], []

    # case-insensitive match against canonical L2 names
    l2_norm = {k.lower(): v for k, v in NAME_LABEL_L2.items()}
    missing = [n for n in l2_names if n.lower() not in l2_norm]
    if missing:
        raise ValueError(f"Unknown L2 names: {missing}. Expected one of: {list(NAME_LABEL_L2.keys())}")

    l2_ids = {l2_norm[n.lower()] for n in l2_names}

    l3_pairs = [
        (l3_name, l3_id)
        for l3_name, (l3_id, l2_id) in REASSIGN_NAME_LABEL_L3L2.items()
        if l2_id in l2_ids
    ]
    l3_pairs.sort(key=lambda x: x[1])  # stable order by L3 id

    l3_names = [n for n, _ in l3_pairs]
    l3_ids = [i for _, i in l3_pairs]
    return l3_names, l3_ids


def l3_values_to_ids(values):
    """
    Convert L3 subset values (names or ids) to ordered L3 ids + names.
    Accepts ints (ids) and strings (names). Numeric strings are treated as ids.
    """
    if not values:
        return [], []

    if isinstance(values, (str, int)):
        values = [values]

    l3_name_to_id = {k.lower(): v for k, v in REASSIGN_NAME_LABEL_L3.items()}
    l3_id_to_name = {v: k for k, v in REASSIGN_NAME_LABEL_L3.items()}

    l3_ids = []
    missing_names = []
    for v in values:
        if isinstance(v, int):
            l3_ids.append(v)
            continue
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.isdigit():
                l3_ids.append(int(v_str))
                continue
            key = v_str.lower()
            if key in l3_name_to_id:
                l3_ids.append(l3_name_to_id[key])
            else:
                missing_names.append(v)
            continue
        raise ValueError(f"Unsupported L3 subset value type: {type(v)} ({v})")

    if missing_names:
        raise ValueError(
            f"Unknown L3 names: {missing_names}. Expected one of: {list(REASSIGN_NAME_LABEL_L3.values())}"
        )

    bad_ids = [i for i in l3_ids if i not in l3_id_to_name]
    if bad_ids:
        raise ValueError(
            f"Unknown L3 ids: {bad_ids}. Expected 0..{max(l3_id_to_name.keys())}"
        )

    l3_ids = sorted(set(l3_ids))
    l3_names = [l3_id_to_name[i] for i in l3_ids]
    return l3_names, l3_ids


def build_l3_to_l2_map():
    """
    Build L3->L2 id mapping and ordered L2 names.

    Returns:
        l3_to_l2: list[int] indexed by L3 id -> L2 id
        l2_names: list[str] indexed by L2 id -> name
    """
    # Order L2 names by their numeric id.
    l2_names = [name for name, _ in sorted(NAME_LABEL_L2.items(), key=lambda kv: kv[1])]

    # Order L3 names by their L3 id, then map to L2 id.
    l3_pairs = sorted(REASSIGN_NAME_LABEL_L3L2.items(), key=lambda kv: kv[1][0])
    l3_to_l2 = [int(l2_id) for _, (_, l2_id) in l3_pairs]

    return l3_to_l2, l2_names
    
GRASSLAND_L3_ATTRS = {
    "Improved Grassland": {
        "vegetation_height": "short to medium vegetation height",
        "sward_texture": "very even close-cropped sward",
        "dominant_cover": "grass-dominated",
        "forb_richness": "few forbs",
    },
    "Neutral Grassland": {
        "vegetation_height": "medium to tall vegetation height",
        "sward_texture": "mixed uneven meadow sward",
        "dominant_cover": "herbs-dominated",
        "forb_richness": "moderate to high forbs",
    },
    "Calcareous Grassland": {
        "vegetation_height": "short vegetation height",
        "sward_texture": "close-cropped open turf",
        "dominant_cover": "fine grasses and herbs dominated",
        "forb_richness": "high forb richness",
    },
    "Acid Grassland": {
        "vegetation_height": "short to medium vegetation height",
        "sward_texture": "patchy or tussocky sward",
        "dominant_cover": "fine grasses dominated",
        "forb_richness": "low to moderate forbs",
    },
    "Bracken": {
        "vegetation_height": "tall vegetation height",
        "sward_texture": "dense canopy of fronds",
        "dominant_cover": "bracken fronds",
        "forb_richness": "low forb richness",
    },
}

# Descriptive attributes for Wetland (L2) L3 classes.
WETLAND_L3_ATTRS = {
    "Fen, Marsh, Swamp": {
        "vegetation_structure": "tall emergent wetland herbs and sedges",
        "dominant_cover": "sedges, rushes, reeds and wetland herbs",
        "surface_texture": "dense emergent cover with wet channels or patches",
        "water_level": "waterlogged to shallowly inundated",
    },
    "Bog": {
        "vegetation_structure": "low open mossy vegetation with scattered dwarf shrubs",
        "dominant_cover": "bog-moss and cotton-grass",
        "surface_texture": "hummocky surface with small wet hollows",
        "water_level": "persistently waterlogged",
    },
}

# Descriptive attributes for Heathland and Shrub (L2) L3 classes.
HEATHLAND_L3_ATTRS = {
    "Dwarf Shrub Heath": {
        "vegetation_height": "low dwarf shrubs (<1.5 m)",
        "vegetation_structure": "dwarf-shrub dominated, low woody canopy",
        "dominant_cover": "heather/ericoids and dwarf gorse",
        "surface_texture": "patchy heather with moss/lichen and bare ground",
    },
}

# Descriptive attributes for Cropland (L2) L3 classes.
CROPLAND_L3_ATTRS = {
    "Arable and Horticulture": {
        "vegetation_structure": "regular planted rows or plots with uniform spacing",
        "dominant_cover": "arable crops or horticultural plantings",
        "surface_texture": "tilled or ploughed soil with furrows and stubble",
        "management_cue": "actively cultivated or rotational fallow",
    },
}

# Descriptive attributes for Woodland and Forest (L2) L3 classes.
WOODLAND_L3_ATTRS = {
    "Broadleaved Mixed and Yew Woodland": {
        "canopy_structure": "tall broadleaved canopy, irregular and layered",
        "foliage_type": "broad leaves with some evergreen yew",
        "understory_light": "dappled light through mixed canopy",
        "ground_cover": "leaf-littered forest floor",
    },
    "Coniferous Woodland": {
        "canopy_structure": "tall conifer canopy, often uniform or plantation-like",
        "foliage_type": "needle-leaved evergreen conifers",
        "understory_light": "darker, more shaded understory",
        "ground_cover": "needle litter with sparse ground vegetation or moss",
    },
}

# Descriptive attributes for Marine Inlets and Transitional Waters (L2) L3 classes.
MARINE_L3_ATTRS = {
    "Littoral Rock": {
        "substrate_type": "exposed rock platforms or boulder shores",
        "surface_texture": "hard, uneven rock with crevices and pools",
        "dominant_cover": "bare rock with algal and barnacle encrustation",
        "tidal_influence": "intertidal, regularly wetted and exposed",
    },
    "Littoral Sediment": {
        "substrate_type": "sand, mud or gravel flats",
        "surface_texture": "flat, soft sediment with ripples",
        "dominant_cover": "mostly bare sediment with sparse algal film",
        "tidal_influence": "intertidal flats, regularly inundated and exposed",
    },
}

# Descriptive attributes for Montane (L2) L3 classes.
MONTANE_L3_ATTRS = {
    "Montane": {
        "vegetation_structure": "low wind-clipped vegetation above treeline",
        "dominant_cover": "dwarf shrubs with moss, lichen and short grasses",
        "surface_texture": "rocky ground with thin soils and bare patches",
        "exposure_cue": "open, treeless, exposed upland ridges",
    },
}

# Descriptive attributes for Rivers and Lakes (L2) L3 classes.
RIVERS_L3_ATTRS = {
    "Standing Open Waters and Canals": {
        "water_body_form": "open water body or straight canal",
        "water_surface": "still or slow-moving open water",
        "bank_structure": "defined banks or engineered canal edges",
        "aquatic_vegetation": "floating or submerged plants with narrow fringe",
    },
}

# Descriptive attributes for Sparsely Vegetated Land (L2) L3 classes.
SPARSE_L3_ATTRS = {
    "Inland Rock": {
        "substrate_type": "exposed inland rock, cliffs or scree",
        "surface_texture": "hard rock faces with fissures and ledges",
        "dominant_cover": "mostly bare rock with sparse crevice plants",
        "exposure_cue": "dry, wind-exposed inland slopes",
    },
    "Supra-littoral Rock": {
        "substrate_type": "coastal rock above the high-tide line",
        "surface_texture": "rugged rock with spray-wet surfaces",
        "dominant_cover": "salt-tolerant lichens or algae, sparse vegetation",
        "exposure_cue": "wave-splash zone with salt spray",
    },
    "Supra-littoral Sediment": {
        "substrate_type": "coastal sand, shingle or pebbles",
        "surface_texture": "loose granular sediment with ridges",
        "dominant_cover": "sparse salt-tolerant pioneer plants",
        "exposure_cue": "above high tide, exposed to spray and wind",
    },
}

# Descriptive attributes for Urban (L2) L3 classes.
URBAN_L3_ATTRS = {
    "Urban": {
        "built_form": "dense built structures, walls and roofs",
        "surface_material": "sealed hard surfaces like concrete or asphalt",
        "vegetation_cover": "little vegetation or small landscaped patches",
        "infrastructure_cue": "roads, kerbs, fences or utilities",
    },
}

# Descriptive attributes for Sea (L2) L3 classes.
SEA_L3_ATTRS = {
    "Sea": {
        "water_body_form": "open marine water to the horizon",
        "surface_texture": "rolling waves or choppy surface",
        "dominant_cover": "open water with minimal vegetation",
        "coastal_context": "distant coastline or open sea view",
    },
}