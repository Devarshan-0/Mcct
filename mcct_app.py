# --- MCCT Model Function ---
def run_mcct_model(plants, features, contexts, num_days, learning_rate, external_df=None):
    import streamlit as st
    import pandas as pd
    
    # --- 1) Prepare Plant Data ---
    np.random.seed(42)
    plant_data = {}

    if external_df is not None:
        # Use uploaded data
        for plant in plants:
            plant_data[plant] = {}
            for day in range(num_days):
                plant_data[plant][day] = {}
                for ctx in contexts:
                    row = external_df[
                        (external_df['plant'] == plant) &
                        (external_df['day'] == day) &
                        (external_df['context'] == ctx)
                    ]
                    if not row.empty:
                        values = row.iloc[0]
                        plant_data[plant][day][ctx] = {
                            f: float(values[f]) for f in features
                        }
                    else:
                        plant_data[plant][day][ctx] = {
                            f: 0 for f in features
                        }
    else:
        # 🆕 Let user edit the table before simulation
        st.info("No CSV uploaded. You can edit the plant data manually below.")

        rows = []
        for plant in plants:
            for day in range(num_days):
                for ctx in contexts:
                    row = {
                        'plant': plant,
                        'day': day,
                        'context': ctx,
                        'temperature': round(np.random.uniform(20, 35), 1),
                        'humidity': round(np.random.uniform(40, 80), 1),
                        'soil_pH': round(np.random.uniform(5.5, 7.5), 2),
                    }
                    rows.append(row)
        
        editable_df = pd.DataFrame(rows)
        edited_df = st.data_editor(editable_df, use_container_width=True, num_rows="dynamic")
        
        for plant in plants:
            plant_data[plant] = {}
            for day in range(num_days):
                plant_data[plant][day] = {}
                for ctx in contexts:
                    row = edited_df[
                        (edited_df['plant'] == plant) &
                        (edited_df['day'] == day) &
                        (edited_df['context'] == ctx)
                    ]
                    if not row.empty:
                        values = row.iloc[0]
                        plant_data[plant][day][ctx] = {
                            f: float(values[f]) for f in features
                        }

    # --- 2) Build Similarity Tensor ---
    n = len(plants)
    m = len(contexts)
    tensor = np.zeros((n, n, num_days, m))
    for i in range(n):
        for j in range(n):
            for d in range(num_days):
                for c, ctx in enumerate(contexts):
                    sim = 0
                    for f in features:
                        vi = plant_data[plants[i]][d][ctx][f]
                        vj = plant_data[plants[j]][d][ctx][f]
                        sim += 1 - abs(vi - vj) / (max(vi, vj) + 1e-5)
                    tensor[i, j, d, c] = sim / len(features)

    # --- 3) Bayesian Causal Inference ---
    influence_prob = np.full((n, n, m), 0.5)

    def compute_causal_matrix(cidx):
        cm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                count = 0
                for d in range(1, num_days):
                    delta = abs(tensor[i, j, d, cidx] - tensor[i, j, d - 1, cidx])
                    if delta > 0.15:
                        count += 1
                cm[i, j] = count / (num_days - 1)
        return cm

    for cidx in range(m):
        cm = compute_causal_matrix(cidx)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                prior = influence_prob[i, j, cidx]
                evidence = cm[i, j]
                influence_prob[i, j, cidx] = prior + learning_rate * (evidence - prior)

    return tensor, influence_prob, plant_data



# --- Streamlit App Starts Here ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import networkx as nx
    # ---- Step 12 Plotting Helpers: Daily heatmap & plant timeseries
# ---- Step 12 Plotting Helpers: Daily heatmap & plant timeseries
def plot_temporal_influence_matrix(day_index):
    matrix = daily_influence_matrices[day_index]
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                cbar_kws={'label': 'Influence'})
    plt.title(f"Influence Matrix - Day {day_index}")
    plt.xlabel("Target Plant")
    plt.ylabel("Source Plant")
    st.pyplot(plt)

def plot_plant_influence_timeseries(plant_index, anomaly_days=None, anomaly_types=None):
    influence_given = []
    influence_received = []

    for mat in daily_influence_matrices:
        influence_given.append(np.sum(mat[plant_index, :]))
        influence_received.append(np.sum(mat[:, plant_index]))

    days = list(range(num_time_steps))
    plt.figure(figsize=(8, 5))
    plt.plot(days, influence_given, label='Influence Given', marker='o')
    plt.plot(days, influence_received, label='Influence Received', marker='s')

    # Draw color-coded anomaly markers
    if anomaly_days and anomaly_types:
        for idx, d in enumerate(anomaly_days.get(plant_index, [])):
            typ = anomaly_types[plant_index][idx]
            color = 'red' if typ == 'increase' else 'blue'
            plt.axvline(x=d, color=color, linestyle='--', alpha=0.6)

    plt.title(f"Temporal Influence for Plant {plant_index}")
    plt.xlabel("Day")
    plt.ylabel("Total Influence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)



# ---- Step 13: Detect sudden influence shifts ----
def detect_influence_shifts(threshold=0.2):
    anomaly_days = {i: [] for i in range(num_plants)}
    anomaly_types = {i: [] for i in range(num_plants)}  # 'increase' or 'decrease'

    for plant_index in range(num_plants):
        prev_given = None
        for day, mat in enumerate(daily_influence_matrices):
            total_given = np.sum(mat[plant_index, :])
            if prev_given is not None:
                change = total_given - prev_given
                if abs(change) > threshold:
                    anomaly_days[plant_index].append(day)
                    anomaly_types[plant_index].append('increase' if change > 0 else 'decrease')
            prev_given = total_given

    return anomaly_days, anomaly_types

def plot_influence_network(matrix, plant_labels=None, threshold=0.1):
    G = nx.DiGraph()

    num_nodes = matrix.shape[0]
    labels = plant_labels if plant_labels else list(range(num_nodes))

    # Add nodes
    for i in range(num_nodes):
        G.add_node(labels[i])

    # Add edges above threshold
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and matrix[i, j] > threshold:
                G.add_edge(labels[i], labels[j], weight=matrix[i, j])

    # Draw layout
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm_weights = [w * 5 for w in weights]  # Scale up for visibility

    plt.figure(figsize=(7, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen',
            node_size=800, arrows=True, edge_color='gray',
            width=norm_weights)
    nx.draw_networkx_edge_labels(G, pos,
            edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()},
            font_color='red')
    plt.title("Causal Influence Network")
    plt.tight_layout()
    st.pyplot(plt)

def get_top_influencers(matrix, plant_labels):
    totals = {}
    for i in range(matrix.shape[0]):
        totals[plant_labels[i]] = np.sum(matrix[i, :])
    sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return sorted_totals
   

# --- Page Setup ---
st.set_page_config(page_title="MCCT Plant Communication", layout="centered")
st.title("🌱 MCCT: Plant Communication Simulator")
st.markdown("Simulate plant-to-plant signaling under environmental contexts using your MCCT model.")

# --- Fixed Parameters ---
# --- Define Plants and Features ---
num_plants = 5
plants = [f"P{i+1}" for i in range(num_plants)]
features = ['temperature', 'humidity', 'soil_pH']
contexts = ['drought', 'normal','stress']
num_contexts = len(contexts)
num_time_steps = 7
learning_rate = 0.2
# --- Sidebar Inputs ---
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Plant Data", type=["csv"])
selected_context = st.sidebar.selectbox("Select Environmental Context", contexts)
selected_time = st.sidebar.slider("Select Time Step", 0, 6, 0)
run_sim = st.sidebar.checkbox("▶️ Run Simulation")


if run_sim:
    st.success("Running real MCCT model...")

    # --- Run Your Real Model ---
    import pandas as pd
    external_df = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            try:
                external_df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("🚫 The uploaded CSV is empty.")
            except Exception as e:
                st.error(f"❌ Could not read the CSV file: {e}")
        else:
            st.error("⚠️ Please upload a file with a `.csv` extension.")

    tensor, influence_prob,plant_data = run_mcct_model(plants, features, contexts, num_time_steps, learning_rate, external_df)
    st.markdown("## 📊 MCCT Simulation Results")

    # Create tabs for each analysis view
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Avg Influence Matrix",
        "📈 Daily Matrix View",
        "📉 Timeseries Trends",
        "🏆 Influence Rankings",
        "🌱 Environmental Data"
    ])

    with tab1:
        st.subheader("📊 Average Influence Matrix (Bayesian Inference)")
        st.write(f"Context: {selected_context}")
        matrix = np.mean(influence_prob[:, :, ctx_idx], axis=1)
        fig1, ax1 = plt.subplots()
        sns.heatmap(matrix, annot=True, cmap="YlGnBu", xticklabels=plants, yticklabels=plants, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        st.subheader("📈 Influence Matrix for Selected Day")
        selected_day = st.selectbox("Select Day", list(range(num_time_steps)), key="matrix_day")
        matrix = influence_prob[:, :, ctx_idx]
        fig2, ax2 = plt.subplots()
        sns.heatmap(matrix[:, selected_day], annot=True, cmap="YlOrRd", xticklabels=plants, yticklabels=plants, ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("📉 Temporal Influence Trends")
        selected_plant = st.selectbox("Select Plant", plants, key="plant_timeseries")
        plant_index = plants.index(selected_plant)
        influence_given = []
        influence_received = []

        for mat in daily_influence_matrices:
            influence_given.append(np.sum(mat[plant_index, :]))
            influence_received.append(np.sum(mat[:, plant_index]))

        days = list(range(num_time_steps))
        fig3, ax3 = plt.subplots()
        ax3.plot(days, influence_given, label='Influence Given', marker='o')
        ax3.plot(days, influence_received, label='Influence Received', marker='s')
        ax3.set_title(f"Temporal Influence for {selected_plant}")
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Total Influence")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

    with tab4:
        st.subheader("🏆 Influence Rankings")
        context_for_ranking = st.selectbox("Select Context for Ranking", contexts, key="ranking_ctx")
        ctx_idx_rank = contexts.index(context_for_ranking)
        avg_influence_matrix = np.mean(tensor[:, :, :, ctx_idx_rank], axis=2)
        total_influence = np.sum(avg_influence_matrix, axis=1)

        fig4, ax4 = plt.subplots()
        bars = ax4.bar(plants, total_influence, color='teal')
        ax4.set_ylabel("Total Outgoing Influence")
        ax4.set_title(f"Overall Influence Rankings — Context: {context_for_ranking}")

        min_val = total_influence.min()
        max_val = total_influence.max()
        margin = (max_val - min_val) * 0.2
        ax4.set_ylim([min_val - margin, max_val + margin])

        for bar, val in zip(bars, total_influence):
            ax4.text(bar.get_x() + bar.get_width()/2, val + margin*0.05, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=9)
        st.pyplot(fig4)

    with tab5:
        st.subheader("🌱 Environmental Data for Each Plant")
        selected_env_day = st.selectbox("Select Day for Environmental View", list(range(num_time_steps)), key="env_day")

        for pi, plant in enumerate(plants):
            try:
                data = plant_data[pi][selected_env_day][selected_context]
                st.markdown(
                    f"**{plant}**: Temperature = {data['temperature']}°C, "
                    f"Humidity = {data['humidity']}%, Soil pH = {data['soil_pH']}"
                )
            except (IndexError, KeyError):
                st.warning(f"No data found for {plant} on Day {selected_env_day} in context {selected_context}.")

        # ---- Step 12: Build daily influence matrices from tensor + Bayesian update
    daily_influence_matrices = []
    for d in range(num_time_steps):
        day_matrix = np.zeros((num_plants, num_plants))
        for i in range(num_plants):
            for j in range(num_plants):
                if i == j:
                    continue
                # replicate your Bayesian update per day
                prior = influence_prob[i, j, contexts.index(selected_context)]
                evidence = tensor[i, j, d, contexts.index(selected_context)]
                influence = prior + learning_rate * (evidence - prior)
                day_matrix[i, j] = influence
        daily_influence_matrices.append(day_matrix)
    avg_influence_matrix = np.mean(np.array(daily_influence_matrices), axis=0)
    
    



    # --- Get Context Index ---
    ctx_idx = contexts.index(selected_context)
    
# --- Day selector for environmental data ---

    st.markdown(f"### 🌿 Environmental Data — Context: {selected_context}, Time Step: {selected_time}")

    for pi, plant in enumerate(plants):
        data = plant_data[plant][selected_time][selected_context]
        st.markdown(
        f"**{plant}**: Temperature = {data['temperature']}°C, "
        f"Humidity = {data['humidity']}%, Soil pH = {data['soil_pH']}"
        )

else:
    st.info("Click ▶️ **Run Simulation** to begin.")
