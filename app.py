import streamlit as st
import pandas as pd
import pulp
import plotly.express as px

# =============================================================================
# 1. Interface Streamlit : Collaborateurs, Poids, Chargement du CSV, Exclusions, Affinités et Tracks Prioritaires
# =============================================================================

st.title("Répartition avec MILP, Slack, contraintes de Tracks, Exclusions, Affinités et Priorités")

# --- 1.1 Collaborateurs ---
if "collaborators" not in st.session_state:
    st.session_state.collaborators = [
        {"Nom": "Louise",    "Contrat": 80,  "Allocation": 0.5},
        {"Nom": "Federica",  "Contrat": 100, "Allocation": 0.5},
        {"Nom": "Alina",     "Contrat": 100, "Allocation": 0.5},
        {"Nom": "Jessica",   "Contrat": 100, "Allocation": 0.5},
        {"Nom": "Vincent",   "Contrat": 100, "Allocation": 0.9},
        {"Nom": "Emilie",    "Contrat": 100, "Allocation": 0.9},
        {"Nom": "Massourou", "Contrat": 100, "Allocation": 0.9},
        {"Nom": "Laura",     "Contrat": 100, "Allocation": 0.9},
        {"Nom": "Ines",      "Contrat": 100, "Allocation": 0.9},
    ]

with st.sidebar:
    st.header("Paramètres de l'application")
    st.subheader("Collaborateurs")
    for idx, collab in enumerate(st.session_state.collaborators):
        cols = st.columns([2, 1, 2, 2])
        with cols[0]:
            name = st.text_input("Nom", value=collab["Nom"], key=f"name_{idx}")
        with cols[1]:
            if st.button("Supprimer", key=f"remove_{idx}"):
                st.session_state.collaborators.pop(idx)
                st.experimental_rerun()
        with cols[2]:
            contrat = st.selectbox(
                "Contrat (%)", 
                options=[50, 80, 100],
                index=[50, 80, 100].index(collab["Contrat"]),
                key=f"contrat_{idx}"
            )
        with cols[3]:
            allocation = st.slider(
                "Allocation", 
                min_value=0.0, 
                max_value=1.0,
                value=collab["Allocation"], 
                step=0.1,
                key=f"alloc_{idx}"
            )
        collab["Nom"] = name
        collab["Contrat"] = contrat
        collab["Allocation"] = allocation

    if st.button("Ajouter collaborateur", key="add_collaborator"):
        st.session_state.collaborators.append({"Nom": "Nouveau collaborateur", "Contrat": 100, "Allocation": 0.5})
        st.experimental_rerun()

    # --- 1.2 Poids des Segments & Tracks ---
    st.header("Poids des Segments (échelle 1 à 3)")
    default_segments = {
        "Self-Paid": 1,
        "Apprenticeship": 1,
        "B2B Pivot & Boost": 1,
        "CPF": 1,
        "Individual funding other": 1,
        "Social Programs - PE FOAD": 1,
        "Social Programs - IDF E-Learning": 1,
        "Social Programs - Other": 1
    }
    segments_weights = {}
    for seg, default_val in default_segments.items():
        segments_weights[seg] = st.slider(
            f"Poids pour segment '{seg}'",
            min_value=1, max_value=5,
            value=default_val,
            step=1,
            key=f"seg_{seg}"
        )

    st.header("Poids des Tracks (échelle 1 à 3)")
    default_tracks = {
        "Systems & Networks": 1,
        "Cybersecurity": 1,
        "Bureautique": 1,
        "Data": 1,
        "Code": 1,
        "Marketing & Comm": 1,
        "Pedagogy": 1,
        "Supply Chain": 1,
        "Business": 1,
        "Design": 1,
        "RH & Gestion": 1,
        "Project Management": 1,
        "Energy": 1,
        "Social & Healthcare": 1,
        "VAE": 1
    }
    tracks_weights = {}
    for track, default_val in default_tracks.items():
        tracks_weights[track] = st.slider(
            f"Poids pour track '{track}'",
            min_value=1, max_value=5,
            value=default_val,
            step=1,
            key=f"track_{track}"
        )

    # --- 1.3 Contraintes sur le nombre de tracks ---
    st.header("Contraintes sur le nombre de tracks")
    min_tracks = st.number_input("Nombre minimum de tracks par collaborateur", value=2, min_value=1, step=1)
    max_tracks = st.number_input("Nombre maximum de tracks par collaborateur", value=6, min_value=1, step=1)

    # --- 1.4 Exclusions par Collaborateur ---
    st.header("Exclusions par Collaborateur")
    exclusion_by_collab = {}
    available_tracks = list(tracks_weights.keys())
    for collab in st.session_state.collaborators:
        exclusion_by_collab[collab["Nom"]] = st.multiselect(
            f"Exclure pour {collab['Nom']}",
            options=available_tracks,
            key=f"excl_{collab['Nom']}"
        )

    # --- 1.5 Affinités de Tracks ---
    st.header("Affinités de Tracks")
    affinity_options = ["Systems & Networks - Cybersecurity", "Code - Bureautique"]
    affinity_selected = st.multiselect(
        "Groupes d'affinité (séparez par ' - ')", 
        options=affinity_options, 
        default=affinity_options
    )
    affinity_groups = []
    for group_str in affinity_selected:
        group = set([t.strip() for t in group_str.split("-")])
        if group:
            affinity_groups.append(group)

    # --- 1.6 Tracks Prioritaires par Collaborateur (Nouvelle fonctionnalité) ---
    st.header("Tracks Prioritaires par Collaborateur")
    priority_by_collab = {}
    for collab in st.session_state.collaborators:
        priority_by_collab[collab["Nom"]] = st.multiselect(
            f"Tracks prioritaires pour {collab['Nom']}",
            options=available_tracks,
            key=f"prio_{collab['Nom']}"
        )

# --- 1.7 Chargement du CSV ---
st.header("Chargement du fichier CSV")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
if not uploaded_file:
    st.info("Veuillez charger un fichier CSV pour continuer.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)
st.success("Fichier CSV chargé avec succès !")
st.dataframe(df.head())

required_cols = ["StudentID_PathID_FinaidID", "Last digit", "Segment", "Track"]
if not all(col in df.columns for col in required_cols):
    st.error(f"Le CSV doit contenir les colonnes {required_cols}.")
    st.stop()

# =============================================================================
# 2. Calcul du temps et groupement (Track, Last digit)
# =============================================================================

def calc_poids(row):
    p_seg = segments_weights.get(row["Segment"], 1)
    p_trk = tracks_weights.get(row["Track"], 1)
    return p_seg * p_trk

df["poids"] = df.apply(calc_poids, axis=1)

# Calcul du temps total disponible pour chaque collaborateur
for c in st.session_state.collaborators:
    c["time_available"] = (c["Contrat"] / 100) * 39 * c["Allocation"] * 3600

total_time_available = sum(c["time_available"] for c in st.session_state.collaborators)
poids_total = df["poids"].sum()
df["temps"] = df["poids"] / poids_total * total_time_available

st.write(f"**Temps total disponible** : {int(total_time_available)} s")

# Groupes (Track, Last digit)
groups_df = df.groupby(["Track", "Last digit"]).agg(
    nb_students=("StudentID_PathID_FinaidID", "count"),
    sum_temps=("temps", "sum")
).reset_index()

# Filtrer les groupes avec nb_students > 0
groups_df = groups_df[groups_df["nb_students"] > 0]

st.write("Aperçu des groupes (Track, Last digit) après filtrage :")
st.dataframe(groups_df.head())

# Initialisation des métriques pour chaque collaborateur
for c in st.session_state.collaborators:
    c["assigned_time"] = 0
    c["assigned_students"] = 0
    c["assigned_groups"] = set()
    c["assigned_tracks"] = set()

# =============================================================================
# 3. Modélisation MILP avec slack, contraintes sur le nombre de tracks,
#    exclusions, affinités et priorités
# =============================================================================
st.header("Attribution via MILP avec contraintes de Tracks, Exclusions, Affinités et Priorités")

# --- 3.1 Définition du problème ---
problem = pulp.LpProblem("Assignment_with_Slack_and_Tracks", pulp.LpMinimize)

# --- 3.2 Variables d'affectation ---
collab_names = [c["Nom"] for c in st.session_state.collaborators]
group_keys = []
group_time = {}
group_students = {}

for _, row in groups_df.iterrows():
    track = row["Track"]
    ld = row["Last digit"]
    g_key = (track, ld)
    group_keys.append(g_key)
    group_time[g_key] = row["sum_temps"]
    group_students[g_key] = row["nb_students"]

# Variables x[g][c] = 1 si le groupe g est attribué au collaborateur c
x = pulp.LpVariable.dicts("x", (group_keys, collab_names), cat=pulp.LpBinary)
# Variables de slack pour chaque collaborateur
slack = pulp.LpVariable.dicts("slack", collab_names, lowBound=0, cat=pulp.LpContinuous)

# --- 3.3 Variables pour la contrainte de tracks ---
tracks = list(groups_df["Track"].unique())
# y[t][c] = 1 si collaborateur c reçoit au moins un groupe du track t
y = pulp.LpVariable.dicts("y", (tracks, collab_names), cat=pulp.LpBinary)
# z[c] = 1 si collaborateur c reçoit au moins un groupe
z = pulp.LpVariable.dicts("z", collab_names, cat=pulp.LpBinary)

# --- 3.4 Objectif : Minimiser la somme des slacks moins un bonus pour les affectations prioritaires ---
epsilon = 0.01  # Petit bonus pour encourager l'affectation des tracks prioritaires
priority_bonus = pulp.lpSum(
    [x[g][c] for c in collab_names for g in group_keys if g[0] in priority_by_collab.get(c, [])]
)
problem += pulp.lpSum([slack[c] for c in collab_names]) - epsilon * priority_bonus, "Minimize_Slack_minus_Priority_Bonus"

# --- 3.5 Contraintes d'affectation ---
for g in group_keys:
    problem += pulp.lpSum([x[g][c] for c in collab_names]) == 1, f"Assign_{g}"

# --- 3.6 Contraintes de capacité + slack ---
capacity = {c["Nom"]: c["time_available"] for c in st.session_state.collaborators}
for c in collab_names:
    problem += pulp.lpSum([group_time[g] * x[g][c] for g in group_keys]) <= capacity[c] + slack[c], f"Cap_{c}"

# --- 3.7 Contraintes sur l'utilisation (détection de collaborateur utilisé) ---
big_M = len(group_keys)
for c in collab_names:
    problem += pulp.lpSum([x[g][c] for g in group_keys]) >= z[c], f"UsedLower_{c}"
    problem += pulp.lpSum([x[g][c] for g in group_keys]) <= big_M * z[c], f"UsedUpper_{c}"

# --- 3.8 Liens entre x et y (pour chaque collaborateur et chaque track) ---
for t in tracks:
    groups_t = [g for g in group_keys if g[0] == t]
    for c in collab_names:
        problem += y[t][c] <= pulp.lpSum([x[g][c] for g in groups_t]), f"Link_y_{t}_{c}"
        M_t = len(groups_t)
        problem += pulp.lpSum([x[g][c] for g in groups_t]) <= M_t * y[t][c], f"Link_y_inv_{t}_{c}"

# --- 3.9 Contraintes de nombre de tracks par collaborateur ---
for c in collab_names:
    problem += pulp.lpSum([y[t][c] for t in tracks]) >= min_tracks * z[c], f"MinTracks_{c}"
    problem += pulp.lpSum([y[t][c] for t in tracks]) <= max_tracks * z[c], f"MaxTracks_{c}"

# --- 3.10 Ajout des exclusions ---
for c in collab_names:
    excluded_tracks = exclusion_by_collab.get(c, [])
    for t in excluded_tracks:
        groups_t = [g for g in group_keys if g[0] == t]
        for g in groups_t:
            problem += x[g][c] == 0, f"Exclude_{g}_{c}"

# --- 3.11 Ajout des affinités ---
for group in affinity_groups:
    tracks_in_group = [t for t in tracks if t in group]
    for c in collab_names:
        if len(tracks_in_group) > 1:
            for t1 in tracks_in_group:
                for t2 in tracks_in_group:
                    if t1 != t2:
                        problem += y[t1][c] == y[t2][c], f"Affinity_{t1}_{t2}_{c}"

# --- 3.12 Résolution avec spinner ---
with st.spinner("Le solveur est en cours d'exécution, veuillez patienter..."):
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60, gapRel=0.01)
    result = problem.solve(solver)

st.write(f"**Statut** : {pulp.LpStatus[result]}")
total_slack_value = pulp.value(problem.objective)
st.write(f"**Somme totale des slacks** : {total_slack_value:.2f}")

# =============================================================================
# 4. Lecture des solutions et mise à jour du DataFrame
# =============================================================================
assignments = {}
for g in group_keys:
    assigned_collab = None
    for c in collab_names:
        if pulp.value(x[g][c]) == 1:
            assigned_collab = c
            break
    if assigned_collab is None:
        assigned_collab = "Non attribué"
    assignments[g] = assigned_collab

def get_assignment(row):
    g_key = (row["Track"], row["Last digit"])
    return assignments.get(g_key, "Non attribué")

df["Collaborateur"] = df.apply(get_assignment, axis=1)

# =============================================================================
# 5. Récapitulatif par collaborateur
# =============================================================================
collab_info = {c: {"assigned_students": 0, "assigned_time": 0.0, "groups": set(), "tracks": set()} for c in collab_names}
for _, row in groups_df.iterrows():
    g_key = (row["Track"], row["Last digit"])
    c_assigned = assignments[g_key]
    if c_assigned != "Non attribué":
        collab_info[c_assigned]["assigned_students"] += row["nb_students"]
        collab_info[c_assigned]["assigned_time"] += row["sum_temps"]
        collab_info[c_assigned]["groups"].add(g_key)
        collab_info[c_assigned]["tracks"].add(g_key[0])  # Track

recap_data = []
for c in st.session_state.collaborators:
    c_name = c["Nom"]
    time_avail = c["time_available"]
    used_time = collab_info[c_name]["assigned_time"]
    used_pct = (used_time / time_avail * 100) if time_avail > 0 else 0
    nb_tracks = len(collab_info[c_name]["tracks"])
    recap_data.append({
        "Collaborateur": c_name,
        "Nb Étudiants": collab_info[c_name]["assigned_students"],
        "Nb Groupes": len(collab_info[c_name]["groups"]),
        "Nb Tracks": nb_tracks,
        "Time dispo (s)": f"{time_avail:.0f}",
        "Temps attribué (s)": f"{used_time:.1f}",
        "% utilisation": f"{used_pct:.1f} %",
        "Slack utilisé": f"{pulp.value(slack[c_name]):.1f}"
    })

recap_df = pd.DataFrame(recap_data)
st.subheader("Récapitulatif de la répartition")
st.dataframe(recap_df)

# =============================================================================
# 6. Histogramme des temps assignable et assigné
# =============================================================================
recap_df["Time dispo (s)"] = recap_df["Time dispo (s)"].str.replace(" s", "").astype(float)
recap_df["Temps attribué (s)"] = recap_df["Temps attribué (s)"].str.replace(" s", "").astype(float)

fig = px.bar(
    recap_df,
    x="Collaborateur",
    y=["Time dispo (s)", "Temps attribué (s)"],
    barmode="group",
    title="Temps assignable et temps assigné par collaborateur",
    labels={"value": "Temps (s)", "variable": "Type de temps"}
)
st.plotly_chart(fig)

# =============================================================================
# 7. Résumé global
# =============================================================================
total_students = len(df)
non_attribues = sum(df["Collaborateur"] == "Non attribué")
pct_non_attribues = 100 * non_attribues / total_students if total_students > 0 else 0

total_groups = len(group_keys)
assigned_groups = sum(assignments[g] != "Non attribué" for g in group_keys)
pct_assigned_groups = 100 * assigned_groups / total_groups if total_groups > 0 else 0

st.write("**Résumé global**")
st.write(f"Étudiants : Total = {total_students}, Non attribués = {non_attribues} ({pct_non_attribues:.2f}%)")
st.write(f"Last Digits (groupes) : Total = {total_groups}, Assignés = {assigned_groups} ({pct_assigned_groups:.2f}%)")

if total_slack_value > 0:
    st.warning(f"La somme des slacks est {total_slack_value:.1f}, indiquant que la solution nécessite un dépassement de capacité.")

# =============================================================================
# 8. Téléchargement des résultats
# =============================================================================
def to_csv_download(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

csv_result = to_csv_download(df)
st.download_button(
    label="Télécharger le CSV avec Collaborateur assigné",
    data=csv_result,
    file_name="repartition_milp_slack_priorites.csv",
    mime="text/csv"
)

st.success("Répartition terminée via MILP avec contraintes de tracks, exclusions, affinités et priorités.")