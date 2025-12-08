import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Configuração para evitar warnings de atribuição
pd.options.mode.chained_assignment = None 

def preprocess(df):
    # Encoding básico de categóricas
    le = LabelEncoder()
    # Copia para não alterar o original fora da função
    df = df.copy()
    for col in ["Sex", "BP", "Cholesterol", "Drug"]:
        if df[col].dtype == 'object':
             df[col] = le.fit_transform(df[col].astype(str))
    return df

# 1. Carrega os dados
df = pd.read_csv('data/kaggle/drug200.csv')

# 2. Pré-processamento Básico (Apenas conversão de strings para números)
df = preprocess(df)

# 3. Define X e y
# Nota: Drug já virou número no preprocess, então y já serve como y_codes
features = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
X = df[features].values
y = df["Drug"].values

# 4. Train/Test Split
# Importante: O split ocorre ANTES de imputar médias ou normalizar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Tratamento de Nulos e Escalonamento (Pipeline Manual)

# Imputação: Aprende a mediana no TREINO e aplica em TREINO e TESTE
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Escalonamento: Aprende a escala no TREINO e aplica em TREINO e TESTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- CORREÇÃO AQUI ---
# Para visualização, precisamos aplicar as transformações aprendidas no treino
# em TODO o conjunto de dados (X). Isso não gera vazamento, pois os parâmetros
# (média, desvio, mediana) vieram apenas do treino.
X_full_imputed = imputer.transform(X)       # Aplica a mediana do treino em tudo
X_scaled_all = scaler.transform(X_full_imputed) # Aplica a escala do treino em tudo
# ---------------------

# 6. PCA
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
# Projeta todos os dados no espaço 2D criado pelo treino
X_pca_all = pca.transform(X_scaled_all)

# 7. Figura e Loop de Kernels
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    "linear": ax1,
    "sigmoid": ax2,
    "poly": ax3,
    "rbf": ax4,
}

results = []

# Como y já é numérico devido ao LabelEncoder, usamos ele para colorir
y_codes = y 

for k, ax in kernels.items():
    # Modelo para métricas (treinado em dados originais escalonados - 5D)
    svm_full = SVC(kernel=k, C=1, random_state=42)
    svm_full.fit(X_train_scaled, y_train)

    y_pred = svm_full.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({"kernel": k, "accuracy": acc})

    # Modelo para visualização (treinado em dados PCA - 2D)
    svm_vis = SVC(kernel=k, C=1, random_state=42)
    svm_vis.fit(X_pca_all, y) # Treina com tudo em 2D apenas para desenhar as fronteiras bonitas

    # Fronteira de decisão
    DecisionBoundaryDisplay.from_estimator(
        svm_vis,
        X_pca_all,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        ax=ax,
    )

    # Plot dos pontos reais
    ax.scatter(
        X_pca_all[:, 0],
        X_pca_all[:, 1],
        c=y_codes,
        s=20,
        edgecolors="k",
        cmap="viridis" # Adicionado colormap para garantir cores distintas
    )

    ax.set_title(k)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# 8. Tabelas e Métricas
results_df = pd.DataFrame(results)

print("<h3>Acurácia por kernel:</h3>")
print(results_df.to_html(classes="table table-bordered table-striped", border=0, index=False))

best_kernel = results_df.loc[results_df["accuracy"].idxmax(), "kernel"]

# Retreina/Recupera melhor modelo
best_clf = SVC(kernel=best_kernel, C=1, random_state=42)
best_clf.fit(X_train_scaled, y_train)
y_pred_best = best_clf.predict(X_test_scaled)

print(f"<h3>Melhor kernel: {best_kernel}</h3>")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best)}")

report = classification_report(y_test, y_pred_best, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("<h3>Relatório de Classificação:</h3>")
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

cm = confusion_matrix(y_test, y_pred_best, labels=best_clf.classes_)
cm_df = pd.DataFrame(cm, index=best_clf.classes_, columns=best_clf.classes_)

print("<h3>Matriz de Confusão:</h3>")
print(cm_df.to_html(classes="table table-bordered table-striped", border=0))

# 9. Exporta Imagem
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()