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
    avg_influence_matrix = np.mean(np.array(daily_influence_matrices), axis=0)
    
    



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
    st.subheader("View & Download Average Influence Matrix")

    matrix_view_mode = st.radio("Select Matrix View", ["Context Matrix", "Overall Average Matrix"])

    if matrix_view_mode == "Context Matrix":
        ctx_idx = contexts.index(selected_context)
        context_matrix = np.mean(tensor[:, :, :, ctx_idx], axis=2)
        csv = pd.DataFrame(context_matrix).to_csv(index=False)
        st.download_button("Download Context Matrix CSV", csv,
                           file_name=f"influence_matrix_{selected_context}.csv",
                           mime='text/csv')
        st.markdown(f"**Context: {selected_context}**")
        fig, ax = plt.subplots()
        sns.heatmap(context_matrix, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
        st.pyplot(fig)

    else:
        avg_matrix = np.mean(daily_influence_matrices, axis=0)
        csv = pd.DataFrame(avg_matrix).to_csv(index=False)
        st.download_button("Download Average Matrix CSV", csv,
                           file_name="average_influence_matrix.csv",
                           mime='text/csv')
        st.markdown("**Overall Average Influence Matrix Across All Days and Contexts**")
        fig, ax = plt.subplots()
        sns.heatmap(avg_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    # ---- Step 12 UI: Temporal Dynamics ----
    st.subheader("View Daily Influence Matrix")
    selected_day = st.slider("Select Day", 0, num_time_steps - 1, 0)
    plot_temporal_influence_matrix(selected_day)

    st.subheader("Influence Time Series per Plant")
    selected_plant = st.selectbox("Select Plant", plants, key="temporal_plant")
    plant_index = plants.index(selected_plant)

    threshold = st.slider("Select Anomaly Threshold", 0.00, 1.00, 0.05, step=0.01)
    anomaly_days, anomaly_types = detect_influence_shifts(threshold=threshold)
    plot_plant_influence_timeseries(plant_index, anomaly_days, anomaly_types)

    # Show detailed anomaly list
    if anomaly_days[plant_index]:
        st.markdown("### 🔍 Sudden Influence Changes Detected")
        for i, day in enumerate(anomaly_days[plant_index]):
            change_type = anomaly_types[plant_index][i]
            arrow = "🔺 Increase" if change_type == "increase" else "🔻 Decrease"
            st.write(f"Day {day}: {arrow}")
    else:
        st.info("No sudden influence shifts detected for this plant.")

    st.subheader("Causal Influence Network")

    network_mode = st.radio("View network for:", ["Average Influence", "Specific Day"])
    network_threshold = st.slider("Edge Threshold", 0.0, 1.0, 0.1, step=0.01)

    if network_mode == "Average Influence":
        plot_influence_network(avg_influence_matrix, plant_labels=plants, threshold=network_threshold)
    else:
        selected_network_day = st.slider("Select Day for Network", 0, num_time_steps - 1, 0)
        plot_influence_network(daily_influence_matrices[selected_network_day],
                               plant_labels=plants, threshold=network_threshold)
    st.subheader("📊 Influence Rankings")

    ranking_mode = st.radio("Select View Mode", ["Top from Network", "Context-Wise Bar Chart"])

    if ranking_mode == "Top from Network":
        if network_mode == "Average Influence":
            ranked = get_top_influencers(avg_influence_matrix, plants)
        else:
            ranked = get_top_influencers(daily_influence_matrices[selected_network_day], plants)

        for rank, (plant, total) in enumerate(ranked, start=1):
            st.write(f"{rank}. {plant} — Total Influence Given: {total:.3f}")

    else:
        # Bar chart version from bottom of your file (merged here)
        st.write("Total Influence Given by Each Plant in Each Context")
        influence_by_context = {
            context: np.sum(tensor[:, :, :, c], axis=(1, 2))
            for c, context in enumerate(contexts)
        }

        fig10, ax10 = plt.subplots(figsize=(10, 5))
        bar_width = 0.2
        positions = np.arange(len(plants))

        for i, (context, values) in enumerate(influence_by_context.items()):
            ax10.bar(positions + i * bar_width, values, width=bar_width, label=context)

        ax10.set_xticks(positions + bar_width * (len(contexts) - 1) / 2)
        ax10.set_xticklabels(plants)
        ax10.set_ylabel("Total Influence")
        ax10.set_title("Influence Given Per Context")
        ax10.legend()
        st.pyplot(fig10)



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
    
    with st.expander("🔍 Pairwise Influence Exploration", expanded=False):

        st.subheader("Feature Trend Visualization")
        feature_to_plot = st.selectbox("Select Feature", ["temperature", "humidity", "pH"])
        for pid in range(num_plants):
            if pid >= len(plant_data):
                continue
            try:
                feature_vals = [plant_data[pid][day][feature_to_plot] for day in range(num_time_steps)]
            except (IndexError, KeyError):
                feature_vals = [0] * num_time_steps  # or np.nan

            plt.plot(range(num_time_steps), feature_vals, label=f"P{pid+1}")
        plt.xlabel("Day")
        plt.ylabel(feature_to_plot.capitalize())
        plt.title(f"{feature_to_plot.capitalize()} Trend Over Time")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Influence Over Time Between Two Plants")
        source = st.selectbox("Select Source Plant", plants, key="src_timewise")
        target = st.selectbox("Select Target Plant", plants, key="tgt_timewise")
        src_idx = plants.index(source)
        tgt_idx = plants.index(target)

        influence_vals = [daily_influence_matrices[d][src_idx, tgt_idx] for d in range(num_time_steps)]
        plt.plot(range(num_time_steps), influence_vals, marker='o')
        plt.title(f"Influence from {source} to {target} Over Time")
        plt.xlabel("Day")
        plt.ylabel("Influence")
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Context-wise Influence Between Two Plants")
        source_ctx = st.selectbox("Select Source Plant", plants, key="src_ctxwise")
        target_ctx = st.selectbox("Select Target Plant", plants, key="tgt_ctxwise")
        src_i = plants.index(source_ctx)
        tgt_i = plants.index(target_ctx)

        influence_by_context = []
        for c in range(len(contexts)):
            context_avg = np.mean(tensor[src_i, tgt_i, :, c])
            influence_by_context.append(context_avg)

        fig9, ax9 = plt.subplots()
        ax9.bar(contexts, influence_by_context, color='orchid')
        ax9.set_ylabel("Influence Value")
        ax9.set_title(f"Influence from {source_ctx} to {target_ctx} by Context")
        st.pyplot(fig9)

    
   

    
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
