#!/usr/bin/env python3
"""
Análise de Similaridade de Projetos - Lei do Bem (Lei 11.196/2005)

Calcula a similaridade par-a-par entre projetos dentro de cada planilha,
com base nos critérios da Lei do Bem, Manual Frascati e Manual Lei do Bem 2025.

Dimensões de análise (com pesos baseados na relevância para P,D&I):
  - Descrição do projeto (peso 0.25): visão geral dos objetivos e escopo
  - Elemento tecnológico (peso 0.20): inovação e avanço tecnológico
  - Desafio tecnológico (peso 0.20): barreiras técnicas enfrentadas
  - Metodologia (peso 0.10): abordagem de pesquisa/desenvolvimento
  - Palavras-chave (peso 0.10): termos técnicos compartilhados (Jaccard)
  - Tipo PB/PA/DE (peso 0.05): classificação Frascati
  - Natureza (peso 0.05): produto vs processo
  - Área do projeto (peso 0.05): setor/área de aplicação
"""

import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==============================================================================
# Configuração
# ==============================================================================
BASE_DIR = "/home/lbarbedo/Projetos/analise_MCTI"
PARTES_DIR = os.path.join(BASE_DIR, "partes_eletroeletronica_2021")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_similaridade")

# Colunas relevantes das planilhas
COL_MAP = {
    "daproj_noprojeto": "nome_projeto",
    "daproj_dsprojeto": "descricao",
    "daproj_tppbpade": "tipo_pdbi",
    "daproj_dsareaprojeto": "area",
    "daproj_dspalavrachave": "palavras_chave",
    "daproj_tpnatureza": "natureza",
    "daproj_dselementotecnologico": "elemento_tecnologico",
    "daproj_dsdesafiotecnologico": "desafio_tecnologico",
    "daproj_dsmetodologiautilizada": "metodologia",
    "lst_norazaosocial": "empresa",
    "lst_nrcnpj": "cnpj",
    "iddadoanaliseprojeto": "id_projeto",
}

# Pesos para cada dimensão (soma = 1.0)
PESOS = {
    "descricao": 0.25,
    "elemento_tecnologico": 0.20,
    "desafio_tecnologico": 0.20,
    "metodologia": 0.10,
    "palavras_chave": 0.10,
    "tipo_pdbi": 0.05,
    "natureza": 0.05,
    "area": 0.05,
}

# Stopwords em português para TF-IDF
STOPWORDS_PT = [
    "a", "ao", "aos", "as", "até", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "era", "essa", "essas", "esse", "esses",
    "esta", "estas", "este", "estes", "eu", "foi", "foram", "há", "isso",
    "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
    "meus", "minha", "minhas", "muito", "na", "nas", "nem", "no", "nos",
    "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "não",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "qual", "quando", "que", "quem", "se", "sem", "seu", "seus", "sua",
    "suas", "são", "só", "também", "te", "tem", "tendo", "ter", "teu",
    "teus", "ti", "toda", "todas", "todo", "todos", "tu", "tua", "tuas",
    "tudo", "um", "uma", "umas", "uns", "você", "vocês", "vós",
    # Stopwords comuns em textos técnicos de projetos Lei do Bem
    "projeto", "desenvolvimento", "pesquisa", "tecnologia", "sistema",
    "utilização", "através", "forma", "sobre", "sendo", "sido",
    "ainda", "pode", "partir", "bem", "cada", "onde", "durante",
    "após", "além", "assim", "então", "será", "foram", "pode",
    "empresa", "objetivo", "realização", "atividades", "resultados",
]


# ==============================================================================
# Funções auxiliares
# ==============================================================================
def limpar_texto(texto):
    """Normaliza texto para análise."""
    if pd.isna(texto) or texto is None:
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\sáàâãéèêíïóôõúüçñ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extrair_palavras_chave(texto):
    """Extrai conjunto de palavras-chave de um texto separado por vírgulas/ponto-e-vírgulas."""
    if pd.isna(texto) or texto is None:
        return set()
    texto = str(texto).lower()
    # Separar por vírgula, ponto-e-vírgula ou barra
    termos = re.split(r"[;,/]", texto)
    return {t.strip() for t in termos if t.strip()}


def jaccard_similarity(set_a, set_b):
    """Calcula similaridade de Jaccard entre dois conjuntos."""
    if not set_a and not set_b:
        return 0.0
    intersecao = set_a & set_b
    uniao = set_a | set_b
    if not uniao:
        return 0.0
    return len(intersecao) / len(uniao)


def similaridade_categorica(val_a, val_b):
    """Retorna 1.0 se iguais, 0.0 se diferentes (para campos categóricos)."""
    if pd.isna(val_a) or pd.isna(val_b):
        return 0.0
    return 1.0 if str(val_a).strip().lower() == str(val_b).strip().lower() else 0.0


def calcular_similaridade_tfidf(textos):
    """Calcula matriz de similaridade de cosseno usando TF-IDF."""
    textos_limpos = [limpar_texto(t) if t else "" for t in textos]

    # Se todos os textos estão vazios, retornar zeros
    if all(t == "" for t in textos_limpos):
        n = len(textos)
        return np.zeros((n, n))

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_PT,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(textos_limpos)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        n = len(textos)
        sim_matrix = np.zeros((n, n))

    return sim_matrix


def carregar_planilha(filepath):
    """Carrega planilha e retorna DataFrame com colunas renomeadas."""
    df = pd.read_excel(filepath, engine="openpyxl")

    # Renomear colunas
    rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    return df


def analisar_planilha(filepath, parte_num):
    """Analisa similaridade par-a-par dentro de uma planilha."""
    df = carregar_planilha(filepath)
    n = len(df)

    if n < 2:
        print(f"  Planilha {parte_num:02d}: apenas {n} projeto(s), pulando.")
        return None, None

    print(f"  Planilha {parte_num:02d}: {n} projetos → {n*(n-1)//2} comparações par-a-par")

    # --- Similaridade TF-IDF para campos textuais ---
    campos_texto = ["descricao", "elemento_tecnologico", "desafio_tecnologico", "metodologia"]
    matrizes_tfidf = {}
    for campo in campos_texto:
        if campo in df.columns:
            textos = df[campo].fillna("").tolist()
            matrizes_tfidf[campo] = calcular_similaridade_tfidf(textos)
        else:
            matrizes_tfidf[campo] = np.zeros((n, n))

    # --- Resultados par-a-par ---
    resultados = []
    for i, j in combinations(range(n), 2):
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        # Similaridades por dimensão
        sims = {}

        # Campos textuais (TF-IDF)
        for campo in campos_texto:
            sims[campo] = float(matrizes_tfidf[campo][i, j])

        # Palavras-chave (Jaccard)
        kw_i = extrair_palavras_chave(row_i.get("palavras_chave", ""))
        kw_j = extrair_palavras_chave(row_j.get("palavras_chave", ""))
        sims["palavras_chave"] = jaccard_similarity(kw_i, kw_j)

        # Campos categóricos (match exato)
        sims["tipo_pdbi"] = similaridade_categorica(
            row_i.get("tipo_pdbi", ""), row_j.get("tipo_pdbi", "")
        )
        sims["natureza"] = similaridade_categorica(
            row_i.get("natureza", ""), row_j.get("natureza", "")
        )
        sims["area"] = similaridade_categorica(
            row_i.get("area", ""), row_j.get("area", "")
        )

        # Nota final ponderada
        nota_final = sum(PESOS[dim] * sims[dim] for dim in PESOS)

        # Verificar se são da mesma empresa
        mesma_empresa = similaridade_categorica(
            row_i.get("cnpj", ""), row_j.get("cnpj", "")
        )

        resultado = {
            "planilha": f"parte_{parte_num:02d}",
            "projeto_A_id": row_i.get("id_projeto", i + 1),
            "projeto_A_nome": str(row_i.get("nome_projeto", ""))[:100],
            "projeto_A_empresa": str(row_i.get("empresa", ""))[:60],
            "projeto_B_id": row_j.get("id_projeto", j + 1),
            "projeto_B_nome": str(row_j.get("nome_projeto", ""))[:100],
            "projeto_B_empresa": str(row_j.get("empresa", ""))[:60],
            "mesma_empresa": int(mesma_empresa),
            "sim_descricao": round(sims["descricao"], 4),
            "sim_elemento_tec": round(sims["elemento_tecnologico"], 4),
            "sim_desafio_tec": round(sims["desafio_tecnologico"], 4),
            "sim_metodologia": round(sims["metodologia"], 4),
            "sim_palavras_chave": round(sims["palavras_chave"], 4),
            "sim_tipo_pdbi": round(sims["tipo_pdbi"], 4),
            "sim_natureza": round(sims["natureza"], 4),
            "sim_area": round(sims["area"], 4),
            "SIMILARIDADE_FINAL": round(nota_final, 4),
        }
        resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values("SIMILARIDADE_FINAL", ascending=False)

    return df_resultados, df


# ==============================================================================
# Main
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("ANÁLISE DE SIMILARIDADE DE PROJETOS - LEI DO BEM")
    print("Critérios: Lei 11.196/2005 | Manual Frascati 2015 | Manual Lei do Bem 2025")
    print("=" * 80)
    print()
    print("Pesos das dimensões:")
    for dim, peso in PESOS.items():
        print(f"  {dim:25s}: {peso:.2f}")
    print()

    # Listar planilhas
    arquivos = sorted([
        f for f in os.listdir(PARTES_DIR)
        if f.endswith(".xlsx")
    ])

    print(f"Planilhas encontradas: {len(arquivos)}")
    print("-" * 80)

    todos_resultados = []
    resumo_planilhas = []

    for idx, arquivo in enumerate(arquivos, start=1):
        filepath = os.path.join(PARTES_DIR, arquivo)
        df_resultado, df_projetos = analisar_planilha(filepath, idx)

        if df_resultado is None:
            continue

        # Salvar resultado individual
        csv_path = os.path.join(OUTPUT_DIR, f"similaridade_parte_{idx:02d}.csv")
        df_resultado.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")

        todos_resultados.append(df_resultado)

        # Estatísticas
        n_pares = len(df_resultado)
        media = df_resultado["SIMILARIDADE_FINAL"].mean()
        maximo = df_resultado["SIMILARIDADE_FINAL"].max()
        minimo = df_resultado["SIMILARIDADE_FINAL"].min()
        alta_sim = (df_resultado["SIMILARIDADE_FINAL"] >= 0.7).sum()
        media_sim = ((df_resultado["SIMILARIDADE_FINAL"] >= 0.5) & (df_resultado["SIMILARIDADE_FINAL"] < 0.7)).sum()

        resumo_planilhas.append({
            "planilha": f"parte_{idx:02d}",
            "n_projetos": len(df_projetos),
            "n_pares": n_pares,
            "similaridade_media": round(media, 4),
            "similaridade_max": round(maximo, 4),
            "similaridade_min": round(minimo, 4),
            "pares_alta_similaridade_>=0.7": alta_sim,
            "pares_media_similaridade_0.5-0.7": media_sim,
        })

        print(f"    → Média: {media:.4f} | Máx: {maximo:.4f} | "
              f"Pares ≥0.7: {alta_sim} | Pares 0.5-0.7: {media_sim}")

    # Consolidar tudo
    print()
    print("=" * 80)
    print("CONSOLIDAÇÃO")
    print("=" * 80)

    if todos_resultados:
        df_consolidado = pd.concat(todos_resultados, ignore_index=True)
        df_consolidado = df_consolidado.sort_values("SIMILARIDADE_FINAL", ascending=False)

        # Salvar consolidado
        csv_consolidado = os.path.join(OUTPUT_DIR, "similaridade_CONSOLIDADO.csv")
        df_consolidado.to_csv(csv_consolidado, index=False, encoding="utf-8-sig", sep=";")

        # Salvar resumo
        df_resumo = pd.DataFrame(resumo_planilhas)
        csv_resumo = os.path.join(OUTPUT_DIR, "resumo_por_planilha.csv")
        df_resumo.to_csv(csv_resumo, index=False, encoding="utf-8-sig", sep=";")

        total_pares = len(df_consolidado)
        media_geral = df_consolidado["SIMILARIDADE_FINAL"].mean()
        pares_altos = (df_consolidado["SIMILARIDADE_FINAL"] >= 0.7).sum()
        pares_medios = ((df_consolidado["SIMILARIDADE_FINAL"] >= 0.5) & (df_consolidado["SIMILARIDADE_FINAL"] < 0.7)).sum()

        print(f"Total de comparações: {total_pares}")
        print(f"Similaridade média geral: {media_geral:.4f}")
        print(f"Pares com alta similaridade (≥ 0.7): {pares_altos}")
        print(f"Pares com média similaridade (0.5-0.7): {pares_medios}")
        print()

        # Top 20 pares mais similares
        print("TOP 20 PARES COM MAIOR SIMILARIDADE:")
        print("-" * 80)
        top20 = df_consolidado.head(20)
        for _, row in top20.iterrows():
            print(f"  [{row['planilha']}] Nota: {row['SIMILARIDADE_FINAL']:.4f}"
                  f" | Mesma empresa: {'Sim' if row['mesma_empresa'] else 'Não'}")
            print(f"    A: {row['projeto_A_nome'][:70]}")
            print(f"       ({row['projeto_A_empresa'][:50]})")
            print(f"    B: {row['projeto_B_nome'][:70]}")
            print(f"       ({row['projeto_B_empresa'][:50]})")
            print()

        # Distribuição
        print("DISTRIBUIÇÃO DE SIMILARIDADE:")
        print("-" * 80)
        faixas = [
            ("0.0 - 0.1", 0.0, 0.1),
            ("0.1 - 0.2", 0.1, 0.2),
            ("0.2 - 0.3", 0.2, 0.3),
            ("0.3 - 0.4", 0.3, 0.4),
            ("0.4 - 0.5", 0.4, 0.5),
            ("0.5 - 0.6", 0.5, 0.6),
            ("0.6 - 0.7", 0.6, 0.7),
            ("0.7 - 0.8", 0.7, 0.8),
            ("0.8 - 0.9", 0.8, 0.9),
            ("0.9 - 1.0", 0.9, 1.01),
        ]
        for label, low, high in faixas:
            count = ((df_consolidado["SIMILARIDADE_FINAL"] >= low) &
                     (df_consolidado["SIMILARIDADE_FINAL"] < high)).sum()
            pct = count / total_pares * 100
            bar = "█" * int(pct / 2)
            print(f"  {label}: {count:5d} ({pct:5.1f}%) {bar}")

        print()
        print(f"Resultados salvos em: {OUTPUT_DIR}/")
        print(f"  - similaridade_CONSOLIDADO.csv (todos os pares)")
        print(f"  - resumo_por_planilha.csv (resumo estatístico)")
        print(f"  - similaridade_parte_XX.csv (resultados por planilha)")


if __name__ == "__main__":
    main()
