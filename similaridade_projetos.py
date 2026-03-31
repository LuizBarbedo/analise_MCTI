"""
Análise de Similaridade entre Projetos da Lei do Bem
=====================================================
Para cada uma das 19 planilhas, calcula a similaridade par a par
entre os 20 projetos (190 pares por planilha, 3.610 pares no total).

A similaridade é calculada com TF-IDF + Cosine Similarity sobre os
campos descritivos dos projetos, que capturam os critérios da Lei do Bem:
  - Descrição do projeto
  - Elemento tecnológico
  - Desafio tecnológico
  - Palavras-chave
  - Metodologia utilizada
  - Área do projeto
  - Nome do projeto
  - Tipo (PB/PA/DE) e setor

Dependências:
  pip install pandas openpyxl scikit-learn

Uso:
  python3 similaridade_projetos.py
"""

import os
import re
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

PASTA_PLANILHAS = "partes_eletroeletronica_2021"
ARQUIVO_SAIDA   = "similaridade_resultados.xlsx"

# Pesos dos campos (repetir texto = dar mais peso ao campo)
CAMPOS_PESOS = {
    "daproj_dsprojeto":           3,   # Descrição completa do projeto
    "daproj_dselementotecnologico": 2, # Elemento tecnológico
    "daproj_dsdesafiotecnologico":  2, # Desafio tecnológico
    "daproj_dspalavrachave":        2, # Palavras-chave
    "daproj_noprojeto":             1, # Nome do projeto
    "daproj_dsareaprojeto":         1, # Área do projeto
    "daproj_dsmetodologiautilizada": 1, # Metodologia
    "daproj_tppbpade":              1, # Tipo (PB/PA/DE)
    "setor_nosetor":                1, # Setor
}

# Stopwords em português
STOPWORDS_PT = [
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "dem", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles",
    "em", "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta",
    "estas", "este", "estes", "eu", "foi", "foram", "há", "isso", "isto",
    "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "minha",
    "muito", "na", "não", "nas", "nem", "no", "nos", "nós", "num", "numa",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
    "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "só", "sua",
    "suas", "também", "te", "tem", "tendo", "ter", "toda", "todas", "todo",
    "todos", "tu", "tua", "tuas", "tudo", "um", "uma", "umas", "uns", "vai",
    "vem", "você", "vocês", "às", "é", "via", "foi", "ser", "pelo",
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def limpar_texto(texto):
    """Remove caracteres especiais e normaliza espaços."""
    if pd.isna(texto) or str(texto).strip().lower() in ("nan", "none", ""):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záàâãéèêíïóôõúüçñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def construir_texto_projeto(linha):
    """Concatena todos os campos com seus pesos (repetição = peso)."""
    partes = []
    for campo, peso in CAMPOS_PESOS.items():
        valor = limpar_texto(linha.get(campo, ""))
        if valor:
            partes.extend([valor] * peso)
    return " ".join(partes)


def calcular_similaridade(df_planilha):
    """
    Recebe um DataFrame de 20 projetos e retorna:
    - matriz 20x20 de similaridade
    - lista de pares (i, j, score)
    """
    textos = df_planilha.apply(construir_texto_projeto, axis=1).tolist()

    # Projetos sem texto recebem texto vazio; similaridade ficará 0
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_PT,
        ngram_range=(1, 2),   # uni e bigramas
        min_df=1,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(textos)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # Caso todos os textos sejam vazio/stopwords apenas
        import numpy as np
        sim_matrix = np.zeros((len(textos), len(textos)))

    # Pares únicos (i < j)
    pares = []
    for i, j in itertools.combinations(range(len(df_planilha)), 2):
        pares.append((i, j, round(float(sim_matrix[i, j]), 4)))

    return sim_matrix, pares


def identificar_projeto(linha):
    """Retorna uma string curta para identificar o projeto nas tabelas."""
    nome = str(linha.get("daproj_noprojeto", "")).strip()
    id_proj = str(linha.get("iddadoanaliseprojeto", "")).strip()
    empresa = str(linha.get("lst_norazaosocial", "")).strip()[:30]

    if nome and nome.lower() not in ("nan", "none"):
        return f"[{id_proj}] {nome[:60]}"
    elif empresa:
        return f"[{id_proj}] {empresa}"
    else:
        return f"Projeto {id_proj}"


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSAMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    arquivos = sorted([
        f for f in os.listdir(PASTA_PLANILHAS)
        if f.endswith(".xlsx")
    ])

    if not arquivos:
        print("Nenhum arquivo .xlsx encontrado em", PASTA_PLANILHAS)
        return

    print(f"Encontrados {len(arquivos)} arquivos.\n")

    # ── Listas para acumular resultados ──────────────────────────────────────
    todos_pares = []         # lista global de todos os pares
    sheets_matriz = {}       # {nome_aba: df_matriz} para output em matriz

    for idx, arquivo in enumerate(arquivos, 1):
        caminho = os.path.join(PASTA_PLANILHAS, arquivo)
        print(f"[{idx:02d}/{len(arquivos)}] Processando: {arquivo}", end=" ... ")

        df = pd.read_excel(caminho)
        df = df.reset_index(drop=True)
        n = len(df)
        print(f"{n} projetos", end=" → ")

        # Identificadores legíveis
        ids = [identificar_projeto(df.loc[i]) for i in range(n)]

        # Cálculo de similaridade
        sim_matrix, pares = calcular_similaridade(df)
        print(f"{len(pares)} pares calculados")

        # ── Matriz de similaridade (para a aba de matrices) ──────────────────
        df_matriz = pd.DataFrame(sim_matrix, index=ids, columns=ids)
        # Nome da aba: até 31 chars (limite Excel)
        nome_aba = arquivo.replace(".xlsx", "").replace("eletroeletronica_2021_", "")[-31:]
        sheets_matriz[nome_aba] = df_matriz

        # ── Lista de pares ────────────────────────────────────────────────────
        for i, j, score in pares:
            todos_pares.append({
                "planilha":        arquivo,
                "id_projeto_A":    df.loc[i, "iddadoanaliseprojeto"] if "iddadoanaliseprojeto" in df.columns else i,
                "nome_projeto_A":  ids[i],
                "id_projeto_B":    df.loc[j, "iddadoanaliseprojeto"] if "iddadoanaliseprojeto" in df.columns else j,
                "nome_projeto_B":  ids[j],
                "empresa_A":       str(df.loc[i, "lst_norazaosocial"])[:60] if "lst_norazaosocial" in df.columns else "",
                "empresa_B":       str(df.loc[j, "lst_norazaosocial"])[:60] if "lst_norazaosocial" in df.columns else "",
                "tipo_A":          str(df.loc[i, "daproj_tppbpade"]) if "daproj_tppbpade" in df.columns else "",
                "tipo_B":          str(df.loc[j, "daproj_tppbpade"]) if "daproj_tppbpade" in df.columns else "",
                "similaridade":    score,
            })

    # ── Exportar para Excel ───────────────────────────────────────────────────
    df_pares = pd.DataFrame(todos_pares)
    df_pares = df_pares.sort_values("similaridade", ascending=False).reset_index(drop=True)

    print(f"\nSalvando resultados em '{ARQUIVO_SAIDA}' ...", end=" ")
    with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl") as writer:

        # Aba 1: Todos os pares em lista (a mais útil)
        df_pares.to_excel(writer, sheet_name="Todos os Pares", index=False)

        # Aba 2: Pares com alta similaridade (>= 0.5)
        df_altos = df_pares[df_pares["similaridade"] >= 0.5].copy()
        df_altos.to_excel(writer, sheet_name="Alta Similaridade (>=0.5)", index=False)

        # Aba por arquivo: matriz de similaridade
        for nome_aba, df_mat in sheets_matriz.items():
            df_mat.to_excel(writer, sheet_name=nome_aba)

    print("OK")

    # ── Estatísticas resumidas ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMO ESTATÍSTICO")
    print("=" * 60)
    print(f"Total de pares analisados : {len(df_pares):,}")
    print(f"Pares com sim >= 0.8      : {(df_pares['similaridade'] >= 0.8).sum():,}")
    print(f"Pares com sim >= 0.5      : {(df_pares['similaridade'] >= 0.5).sum():,}")
    print(f"Pares com sim >= 0.3      : {(df_pares['similaridade'] >= 0.3).sum():,}")
    print(f"Similaridade média        : {df_pares['similaridade'].mean():.4f}")
    print(f"Similaridade mediana      : {df_pares['similaridade'].median():.4f}")
    print(f"Similaridade máxima       : {df_pares['similaridade'].max():.4f}")

    print("\nTop 10 pares mais similares:")
    print(df_pares[["nome_projeto_A", "nome_projeto_B", "similaridade"]].head(10).to_string(index=False))

    print(f"\nArquivo salvo: {os.path.abspath(ARQUIVO_SAIDA)}")


if __name__ == "__main__":
    main()
