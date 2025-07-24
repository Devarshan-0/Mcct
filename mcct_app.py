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
    plt.show()

def plot_plant_influence_timeseries(plant_index):
    influence_given = []
    influence_received = []

    for mat in daily_influence_matrices:
        influence_given.append(np.sum(mat[plant_index, :]))
        influence_received.append(np.sum(mat[:, plant_index]))

    days = list(range(num_time_steps))
    plt.figure(figsize=(8, 5))
    plt.plot(days, influence_given, label='Influence Given', marker='o')
    plt.plot(days, influence_received, label='Influence Received', marker='s')
    plt.title(f"Temporal Influence for Plant {plant_index}")
    plt.xlabel("Day")
    plt.ylabel("Total Influence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

   




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
external_df = pd.read_csv(uploaded_file) if uploaded_file is not None else None
selected_context = st.sidebar.selectbox("Select Environmental Context", contexts)
selected_time = st.sidebar.slider("Select Time Step", 0, 6, 0)
threshold = st.sidebar.slider("Minimum Influence Threshold", 0.0, 1.0, 0.1, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Time-wise Influence Tracker")
run_sim = st.sidebar.checkbox("▶️ Run Simulation")


if run_sim:
    st.success("Running real MCCT model...")

    # --- Run Your Real Model ---
    import pandas as pd
    external_df = pd.read_csv(uploaded_file) if uploaded_file is not None else None
    tensor, influence_prob,plant_data = run_mcct_model(plants, features, contexts, num_time_steps, learning_rate, external_df)
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



    # --- Get Context Index ---
    ctx_idx = contexts.index(selected_context)

    # --- Show Influence Matrix (from Bayesian) ---
    st.subheader(f"Bayesian Influence Matrix — Context: {selected_context}")
    matrix = influence_prob[:, :, ctx_idx]
    fig1, ax1 = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", xticklabels=plants, yticklabels=plants, ax=ax1)
    st.pyplot(fig1)

    # Convert influence matrix to CSV for download
    df_influence = pd.DataFrame(matrix, columns=plants, index=plants)
    st.download_button(
        label="📥 Download Influence Matrix as CSV",
        data=df_influence.to_csv().encode('utf-8'),
        file_name=f"influence_matrix_{selected_context}.csv",
        mime='text/csv'
    )
       # ---- Step 12 UI: Temporal Dynamics ----
    st.subheader("View Daily Influence Matrix")
    selected_day = st.slider("Select Day", 0, num_time_steps - 1, 0)
    plot_temporal_influence_matrix(selected_day)

    st.subheader("Influence Time Series per Plant")
    selected_plant = st.selectbox("Select Plant", plants, key="temporal_plant")
    plant_index = plants.index(selected_plant)
    plot_plant_influence_timeseries(plant_index)



    # --- PCA Visualization ---
    st.subheader("PCA Projection of Influence Matrix")
    flat = matrix.reshape(num_plants, -1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    fig2, ax2 = plt.subplots()
    ax2.scatter(coords[:, 0], coords[:, 1])
    for i in range(num_plants):
        ax2.annotate(plants[i], (coords[i, 0], coords[i, 1]))
    st.pyplot(fig2)

    # --- Network Graph of Influence ---
    st.subheader("Plant Influence Graph")
    G = nx.DiGraph()
    for i in range(num_plants):
        G.add_node(plants[i])
    for i in range(num_plants):
        for j in range(num_plants):
            if i != j and matrix[i, j] > threshold:
                G.add_edge(plants[i], plants[j], weight=round(matrix[i, j], 2))


    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    fig3, ax3 = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1200, ax=ax3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax3)
    st.pyplot(fig3)
# --- Feature Trend Visualization ---
    st.subheader("📉 Feature Trends Over Time")
    selected_feature = st.selectbox("Select Feature to View", features)

    fig4, ax4 = plt.subplots()
    for plant in plants:
      y = [plant_data[plant][day][selected_context][selected_feature] for day in range(num_time_steps)]
      ax4.plot(range(num_time_steps), y, label=plant, marker='o')
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel(selected_feature.capitalize())
    ax4.set_title(f"{selected_feature.capitalize()} over Time — Context: {selected_context}")
    ax4.legend()
    st.pyplot(fig4)
    st.subheader("🔁 Time-wise Influence Between Two Plants")

    selected_source = st.sidebar.selectbox("Source Plant", plants, key="src_timewise")
    selected_target = st.sidebar.selectbox("Target Plant", plants, key="tgt_timewise")

    if selected_source != selected_target:
        source_idx = plants.index(selected_source)
        target_idx = plants.index(selected_target)

        forward = [tensor[source_idx, target_idx, t, ctx_idx] for t in range(num_time_steps)]
        backward = [tensor[target_idx, source_idx, t, ctx_idx] for t in range(num_time_steps)]

        fig6, ax6 = plt.subplots()
        ax6.plot(range(num_time_steps), forward, marker='o', color='crimson', linestyle='-', label=f"{selected_source} → {selected_target}")
        ax6.plot(range(num_time_steps), backward, marker='s', color='green', linestyle='--', label=f"{selected_target} → {selected_source}")

        ax6.set_xlabel("Time Step")
        ax6.set_ylabel("Influence Score")
        ax6.set_title(f"Influence Over Time — Context: {selected_context}")
        ax6.legend()
        st.pyplot(fig6)

    else:
        st.warning("Please select two different plants to compare their influence.")
    st.subheader("🌍 Context-wise Influence for a Plant Pair")

    context_time_step = st.sidebar.slider("Select Time Step for Context Comparison", 0, num_time_steps - 1, 0, key="context_time_slider")
    context_source = st.sidebar.selectbox("Source Plant (Context-wise)", plants, key="context_source")
    context_target = st.sidebar.selectbox("Target Plant (Context-wise)", plants, key="context_target")

    if context_source != context_target:
        source_idx = plants.index(context_source)
        target_idx = plants.index(context_target)

        influence_by_context = [
            tensor[source_idx, target_idx, context_time_step, i] for i in range(num_contexts)
        ]

        fig7, ax7 = plt.subplots()
        ax7.bar(contexts, influence_by_context, color=['orange', 'green', 'skyblue'])
        ax7.set_ylim(0, 1)
        ax7.set_ylabel("Influence Score")
        ax7.set_title(f"Influence of {context_source} → {context_target} at Time Step {context_time_step}")
        st.pyplot(fig7)
    else:
        st.warning("Please choose two different plants to compare influence.")
    st.subheader("📊 Average Influence Matrix Over Time")

    avg_ctx = st.selectbox("Select Context", contexts, key="avg_ctx_inline")

    ctx_idx_avg = contexts.index(avg_ctx)

    avg_matrix = np.mean(tensor[:, :, :, ctx_idx_avg], axis=2)

    fig8, ax8 = plt.subplots()
    sns.heatmap(avg_matrix, annot=True, cmap="viridis", xticklabels=plants, yticklabels=plants, ax=ax8)
    ax8.set_title(f"Average Influence Over Time — Context: {avg_ctx}")
    st.pyplot(fig8)
    st.subheader("🌐 Feature-wise Influence Matrix")

    selected_feature_influence = st.selectbox("Select Feature", features, key="feature_influence")
    feature_index = features.index(selected_feature_influence)

    matrix_feature_based = np.zeros((num_plants, num_plants))
    for i in range(num_plants):
        for j in range(num_plants):
            if i == j:
                continue
            influence_vals = []
            for t in range(num_time_steps):
                vi = plant_data[plants[i]][t][selected_context][selected_feature_influence]
                vj = plant_data[plants[j]][t][selected_context][selected_feature_influence]
                sim = 1 - abs(vi - vj) / (max(vi, vj) + 1e-5)
                influence_vals.append(sim)
            matrix_feature_based[i, j] = np.mean(influence_vals)

    fig9, ax9 = plt.subplots()
    sns.heatmap(matrix_feature_based, annot=True, cmap="coolwarm", xticklabels=plants, yticklabels=plants, ax=ax9)
    ax9.set_title(f"Feature-based Influence Matrix — {selected_feature_influence.capitalize()} ({selected_context})")
    st.pyplot(fig9)
    st.subheader("🏆 Plant Influence Rankings (Total Outgoing Influence)")

    context_for_ranking = st.selectbox("Select Context for Ranking", contexts, key="ranking_ctx")
    ctx_idx_rank = contexts.index(context_for_ranking)

    avg_influence_matrix = np.mean(tensor[:, :, :, ctx_idx_rank], axis=2)
    total_influence = np.sum(avg_influence_matrix, axis=1)

    fig10, ax10 = plt.subplots()
    bars = ax10.bar(plants, total_influence, color='teal')
    ax10.set_ylabel("Total Outgoing Influence")
    ax10.set_title(f"Overall Influence Rankings — Context: {context_for_ranking}")

    # --- Adjust y-axis scale to zoom into the differences ---
    min_val = total_influence.min()
    max_val = total_influence.max()
    margin = (max_val - min_val) * 0.2 # 20% margin
    ax10.set_ylim([min_val - margin, max_val + margin])

    # --- Annotate bars with exact values ---
    for bar, val in zip(bars, total_influence):
        ax10.text(bar.get_x() + bar.get_width()/2, val + margin*0.05, f"{val:.2f}",
                  ha='center', va='bottom', fontsize=9)

    st.pyplot(fig10)



    

# --- Show Environmental Data ---
    st.markdown(f"### 🌿 Environmental Data — Context: {selected_context}, Time Step: {selected_time}")

    for pi, plant in enumerate(plants):
        data = plant_data[plant][selected_time][selected_context]
        st.markdown(
        f"**{plant}**: Temperature = {data['temperature']}°C, "
        f"Humidity = {data['humidity']}%, Soil pH = {data['soil_pH']}"
        )
else:
    st.info("Click ▶️ **Run Simulation** to begin.")
