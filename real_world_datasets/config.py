# Configuration file for datasets
# Top teams for each dataset
TOP_TEAMS = {
    "basketball": [
        ' Gonzaga             ',
        ' Illinois            ',
        ' Baylor              ',
        ' Michigan            ',
        ' Alabama             ',
        ' Houston             ',
        ' Ohio St             ',
        ' Iowa                ',
        ' Texas               ',
        ' Arkansas            ',
        ' Oklahoma St         ',
        ' Kansas              ',
        ' West Virginia       ',
        ' Florida St          ',
        ' Virginia            '
    ],
    "football": [
        ' Georgia             ',
        ' Alabama             ',
        ' Michigan            ',
        ' Cincinnati          ',
        ' Baylor              ',
        ' Ohio St             ',
        ' Oklahoma St         ',
        ' Notre Dame          ',
        ' Michigan St         ',
        ' Oklahoma            ',
        ' Mississippi         ',
        ' Utah                ',
        ' Pittsburgh          ',
        ' Clemson             ',
        ' Wake Forest         '
    ]
}

# Data file mappings
FILES = {
    "basketball": ["cb2021.csv", "cb2020.csv"],
    "football": ["cf2021.csv", "cf2020.csv", "cf2019.csv"],
    "baseball": ["cb2021.csv", "cb2020.csv", "cb2019.csv"],
} 



MODEL_LABEL     = {'Original L1': 'ML', 'Mallows Footrule': 'footrule', 'Mallows Spearman': 'spearman', 'Plackett-Luce': 'PL', 'Plackett-Luce-Reg': 'PL_reg', 'Kendall': 'kendal'}
MODEL_NICE_NAME = {'Original L1': 'L_α-Mallows', 'Mallows Footrule': 'L₁-Mallows', 'Mallows Spearman': 'L₂-Mallows', 'Plackett-Luce': 'Plackett–Luce',
                'Plackett-Luce-Reg': 'Plackett–Luce (Ridge)', 'Kendall': "Mallows τ"}

METRICS_NICE_NAMES = {                       # key-template,   ↑/↓ label,  display-factor
    'spearman_rho':  ('spearman_rho_{}',   "↑ Spearman's ρ",    1),
    'kendall_tau':   ('kendall_tau_{}',    "↑ Kendall's τ",     1),
    'hamming':       ('hamming_distance_{}', "↓ Hamming distance", 1),
    'pairwise_acc':  ('pairwise_acc_{}',   "↑ Pairwise acc. (%)", 100),
    'top_k':         ('top_k_hit_rates_{}', "↑ Top-{k} hit rate (%)", 100),
}

METRIC_NAMES = (
    "top_k_hit_rates",
    "spearman_rho",
    "hamming_distance",
    "kendall_tau",
    "ndcg",
    "pairwise_acc",
)