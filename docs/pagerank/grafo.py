import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CONFIGURAÇÕES PARA MKDOCS ---
# Caminho onde a imagem será salva (relativo a onde você roda o script)
# Ajuste conforme sua estrutura de pastas do MkDocs
OUTPUT_DIR = "docs/pagerank"
OUTPUT_FILENAME = "grafo_cit_hepth_top50.png"
FULL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Garante que a pasta existe antes de tentar salvar
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Diretório criado: {OUTPUT_DIR}")

# --- 1. CARREGAMENTO ---
print("Carregando dataset Cit-HepTh...")
# O arquivo é separado por TABs (\t) e tem linhas de comentário (#)
# Ajuste o caminho do arquivo .txt se necessário
df = pd.read_csv('data/grafo/Cit-HepTh.txt', sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'], dtype=str)

# Cria o grafo dirigido
G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId', create_using=nx.DiGraph())

print(f"Grafo carregado: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

# --- 2. FILTRAGEM (Subgrafo dos Hubs) ---
# Seleciona os 50 nós com maior grau para a visualização ficar limpa
k = 50
node_degrees = dict(G.degree())
top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:k]
subgraph = G.subgraph(top_nodes)

# --- 3. PLOTAGEM ---
# Tamanho da figura (em polegadas). 10x8 fica bom em telas de PC.
plt.figure(figsize=(10, 8))

# Layout: Spring layout funciona bem para revelar clusters
# seed=42 garante que o desenho seja sempre igual (importante para documentação)
pos = nx.spring_layout(subgraph, seed=42, k=0.6)

# Desenha Nós
# O tamanho do nó depende do grau (mais conexões = nó maior)
degrees = dict(subgraph.degree())
node_sizes = [v * 10 for v in degrees.values()]

nx.draw_networkx_nodes(subgraph, pos, 
                       node_size=node_sizes, 
                       node_color='#4a90e2', # Um azul agradável
                       edgecolors='white',   # Borda branca no nó para contraste
                       linewidths=1.5,
                       alpha=0.9)

# Desenha Arestas
nx.draw_networkx_edges(subgraph, pos, 
                       edge_color='gray', 
                       alpha=0.4, 
                       arrows=True,
                       arrowstyle='-|>', 
                       arrowsize=12,
                       connectionstyle='arc3,rad=0.1') # Curva leve para arestas bidirecionais não se sobreporem

# Desenha Labels (IDs)
nx.draw_networkx_labels(subgraph, pos, 
                        font_size=8, 
                        font_family='sans-serif',
                        font_weight='bold',
                        font_color='#333333')

plt.title(f"Top {k} Papers Mais Conectados (Cit-HepTh)", fontsize=16, pad=20)
plt.axis('off') # Remove eixos X e Y

# --- 4. SALVAR PARA MKDOCS ---
# bbox_inches='tight' remove bordas brancas excessivas
# transparent=True deixa o fundo transparente para integrar com temas dark/light do MkDocs
plt.savefig(FULL_OUTPUT_PATH, format="PNG", dpi=150, bbox_inches='tight', transparent=True)
plt.close() # Fecha a figura para liberar memória

print(f"Imagem salva com sucesso em: {FULL_OUTPUT_PATH}")
print("Agora você pode referenciá-la no seu Markdown.")