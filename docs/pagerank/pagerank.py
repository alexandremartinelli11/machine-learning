import networkx as nx
import pandas as pd
import numpy as np

# --- 1. CONFIGURAÇÃO E LEITURA ---
dataset_file = 'data/grafo/Cit-HepTh.txt'

try:
    df = pd.read_csv(dataset_file, sep='\t', comment='#', names=['From', 'To'], dtype=str)
    G = nx.from_pandas_edgelist(df, 'From', 'To', create_using=nx.DiGraph())

    # --- 2. IMPLEMENTAÇÃO MANUAL CORRIGIDA (MATRICIAL) ---
    def pagerank_manual_v2(G, d=0.85, tol=1.0e-4, max_iter=100):
        # Mapeia nós para índices 0..N-1
        nodes = list(G.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Cria Matriz de Transição Esparsa (M)
        # M_ij = 1/L(j) se j aponta para i, senão 0
        import scipy.sparse as sp
        
        row = []
        col = []
        data = []
        
        dangling_nodes = [] # Nós que não apontam pra ninguém
        
        for node in nodes:
            out_edges = list(G.successors(node))
            if len(out_edges) == 0:
                dangling_nodes.append(node_to_idx[node])
            else:
                weight = 1.0 / len(out_edges)
                for target in out_edges:
                    row.append(node_to_idx[target])
                    col.append(node_to_idx[node])
                    data.append(weight)
        
        M = sp.csr_matrix((data, (row, col)), shape=(n, n))
        
        # Vetor inicial (1/N)
        v = np.ones(n) / n
        
        # Loop de Potência
        for it in range(max_iter):
            v_last = v.copy()
            
            # 1. Aplica a matriz de links: v = d * M * v
            v = d * (M @ v)
            
            # 2. Distribui a massa dos dangling nodes e o fator de teleporte (1-d)
            # Soma dos pesos dos nós sem saída
            dangling_sum = sum(v_last[i] for i in dangling_nodes)
            
            # Fator de correção para garantir soma 1
            # (1-d) + (d * dangling_sum) é distribuído igualmente para todos
            correction = (1.0 - sum(v)) / n
            v += correction
            
            # Checa convergência (Norma L1)
            err = np.sum(np.abs(v - v_last))
            if err < tol:
                break
                
        # Converte de volta para dicionário {ID: Score}
        return {nodes[i]: v[i] for i in range(n)}, it + 1

    # --- 3. EXECUÇÃO E COMPARAÇÃO ---
    pr_manual, iters = pagerank_manual_v2(G, d=0.85)
    pr_nx = nx.pagerank(G, alpha=0.85, tol=1.0e-4)

    # --- 4. GERAÇÃO DE SAÍDA ---
    # Pegamos os Top 10 baseados no NETWORKX (Gabarito) para ver se o manual bate
    top_10_nx = sorted(pr_nx, key=pr_nx.get, reverse=True)[:10]
    
    results = []
    for rank, node in enumerate(top_10_nx, 1):
        v_manual = pr_manual[node]
        v_nx = pr_nx[node]
        
        results.append({
            "Rank (NX)": rank,
            "Paper ID": node,
            "In-Degree": G.in_degree(node),
            "PR Manual": f"{v_manual:.6f}",
            "PR NetworkX": f"{v_nx:.6f}",
            "Dif": f"{abs(v_manual - v_nx):.2e}" # Deve dar algo como 1.0e-16
        })

    df_res = pd.DataFrame(results)

    print(f"<p>Convergência em {iters} iterações.</p>")
    print(df_res.to_html(classes="table table-bordered table-striped", index=False, justify='center', border=0))

except Exception as e:
    print(e)