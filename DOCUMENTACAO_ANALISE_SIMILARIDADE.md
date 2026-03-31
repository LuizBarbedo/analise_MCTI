# Documentacao Tecnica - Analise de Similaridade de Projetos

## 1) Objetivo

Este documento descreve o processo completo de analise de similaridade entre projetos da Lei do Bem, conforme implementado no script `analise_similaridade.py` e refletido nos arquivos CSV de saida em `resultados_similaridade/`.

O foco da analise e identificar pares de projetos com alto grau de sobreposicao tecnica, considerando texto descritivo, classificacoes e metadados de projeto.

---

## 2) Escopo de dados analisados

- Entrada principal: planilhas `.xlsx` em `partes_eletroeletronica_2021/`
- Saidas principais:
  - `resultados_similaridade/similaridade_parte_XX.csv`
  - `resultados_similaridade/similaridade_CONSOLIDADO.csv`
  - `resultados_similaridade/resumo_por_planilha.csv`

### 2.1 Logica de comparacao

A comparacao e feita **par a par** dentro de cada planilha.

Se uma planilha tem `n` projetos, o numero de comparacoes e:

`n * (n - 1) / 2`

Exemplo: para `n = 20`, temos `190` pares.

---

## 3) Mapeamento das colunas

As colunas da planilha original sao renomeadas internamente para nomes padrao:

- `daproj_noprojeto` -> `nome_projeto`
- `daproj_dsprojeto` -> `descricao`
- `daproj_tppbpade` -> `tipo_pdbi`
- `daproj_dsareaprojeto` -> `area`
- `daproj_dspalavrachave` -> `palavras_chave`
- `daproj_tpnatureza` -> `natureza`
- `daproj_dselementotecnologico` -> `elemento_tecnologico`
- `daproj_dsdesafiotecnologico` -> `desafio_tecnologico`
- `daproj_dsmetodologiautilizada` -> `metodologia`
- `lst_norazaosocial` -> `empresa`
- `lst_nrcnpj` -> `cnpj`
- `iddadoanaliseprojeto` -> `id_projeto`

---

## 4) Pre-processamento de texto

Para campos textuais, e aplicada limpeza:

1. Conversao para minusculas
2. Remocao de pontuacao e caracteres especiais
3. Normalizacao de espacos
4. Aplicacao de stopwords em portugues

Stopwords tecnicas adicionais tambem sao removidas (por exemplo: "projeto", "desenvolvimento", "pesquisa", etc.) para reduzir ruido semantico.

---

## 5) Dimensoes de similaridade

Cada par de projetos recebe uma nota parcial por dimensao.

### 5.1 Dimensoes textuais (TF-IDF + Cosseno)

- `sim_descricao`
- `sim_elemento_tec`
- `sim_desafio_tec`
- `sim_metodologia`

Metodo:

1. Vetorizacao TF-IDF (unigramas + bigramas)
2. Similaridade do cosseno entre vetores

Parametros relevantes do vetorizador:

- `ngram_range = (1, 2)`
- `max_features = 5000`
- `min_df = 1`

### 5.2 Palavras-chave (Jaccard)

- `sim_palavras_chave`

Metodo:

1. Quebra dos termos por `,`, `;` e `/`
2. Conversao para conjuntos
3. Calculo de Jaccard:

`J(A, B) = |A intersecao B| / |A uniao B|`

### 5.3 Campos categoricos (match exato)

- `sim_tipo_pdbi`
- `sim_natureza`
- `sim_area`

Regra:

- 1.0 se os valores sao iguais
- 0.0 se os valores sao diferentes

---

## 6) Pesos oficiais da nota final

A nota final e uma media ponderada com soma dos pesos igual a 1.0:

- `descricao`: 0.25
- `elemento_tecnologico`: 0.20
- `desafio_tecnologico`: 0.20
- `metodologia`: 0.10
- `palavras_chave`: 0.10
- `tipo_pdbi`: 0.05
- `natureza`: 0.05
- `area`: 0.05

### 6.1 Formula da nota final

Sejam:

- `D` = similaridade de descricao
- `E` = similaridade de elemento tecnologico
- `T` = similaridade de desafio tecnologico
- `M` = similaridade de metodologia
- `K` = similaridade de palavras-chave
- `P` = similaridade de tipo PB/PA/DE
- `N` = similaridade de natureza
- `A` = similaridade de area

Entao:

`SIMILARIDADE_FINAL = 0.25*D + 0.20*E + 0.20*T + 0.10*M + 0.10*K + 0.05*P + 0.05*N + 0.05*A`

A faixa da nota e `[0, 1]`.

---

## 7) Exemplo de calculo

Suponha um par com:

- `D = 0.80`
- `E = 0.60`
- `T = 0.50`
- `M = 0.90`
- `K = 0.40`
- `P = 1.00`
- `N = 1.00`
- `A = 0.00`

Aplicando a formula:

`SIMILARIDADE_FINAL = 0.25*0.80 + 0.20*0.60 + 0.20*0.50 + 0.10*0.90 + 0.10*0.40 + 0.05*1.00 + 0.05*1.00 + 0.05*0.00`

`SIMILARIDADE_FINAL = 0.20 + 0.12 + 0.10 + 0.09 + 0.04 + 0.05 + 0.05 + 0.00 = 0.65`

Interpretacao: similaridade media-alta.

---

## 8) Criterios de interpretacao (faixas)

No script, as faixas operacionais usadas no resumo sao:

- Alta similaridade: `>= 0.7`
- Media similaridade: `>= 0.5` e `< 0.7`

Notas abaixo de `0.5` indicam menor convergencia tecnica relativa entre os projetos comparados.

---

## 9) Colunas de saida e significado

No consolidado (`similaridade_CONSOLIDADO.csv`), as colunas principais sao:

- Identificacao dos pares:
  - `planilha`
  - `projeto_A_id`, `projeto_A_nome`, `projeto_A_empresa`
  - `projeto_B_id`, `projeto_B_nome`, `projeto_B_empresa`
- Indicador auxiliar:
  - `mesma_empresa` (1 = mesmo CNPJ, 0 = CNPJ diferente)
- Similaridades por dimensao:
  - `sim_descricao`
  - `sim_elemento_tec`
  - `sim_desafio_tec`
  - `sim_metodologia`
  - `sim_palavras_chave`
  - `sim_tipo_pdbi`
  - `sim_natureza`
  - `sim_area`
- Resultado final:
  - `SIMILARIDADE_FINAL`

Observacao importante: `mesma_empresa` **nao entra** no calculo da nota final. E apenas uma informacao de contexto.

---

## 10) Diferenca entre os dois scripts do repositorio

Ha dois scripts com abordagens diferentes:

1. `analise_similaridade.py` (metodo em uso nos CSV atuais)
   - Nota final por dimensao com pesos explicitos
   - Mais interpretavel e auditavel

2. `similaridade_projetos.py` (metodo alternativo)
   - Concatena campos em um texto unico
   - Aplica "peso" por repeticao de texto
   - Gera um score unico de cosseno sem decomposicao por dimensao

Para analise tecnica com justificativa de nota, o metodo de `analise_similaridade.py` e o recomendado.

---

## 11) Limites e cuidados metodologicos

1. Dependencia da qualidade do preenchimento dos campos textuais.
2. Similaridade lexical alta pode ocorrer em projetos administrativos parecidos sem identidade tecnica completa.
3. Campos categoricos com regra binaria (0/1) podem simplificar demais diferencas sutis.
4. Projetos quase duplicados em empresas do mesmo grupo podem elevar a distribuicao de notas altas.

---

## 12) Melhorias futuras sugeridas

1. Lematizacao/stemming em portugues para reduzir variacoes morfologicas.
2. Dicionario de sinonimos tecnicos por dominio setorial.
3. Embeddings semanticos para captar similaridade de significado alem de termos identicos.
4. Recalibracao de pesos com validacao por especialistas.
5. Regra de penalizacao opcional para pares potencialmente duplicados por replicacao empresarial.

---

## 13) Reproducao da analise

No diretorio do projeto:

1. Garantir dependencias instaladas: `pandas`, `openpyxl`, `numpy`, `scikit-learn`
2. Executar:

`python3 analise_similaridade.py`

3. Conferir saidas em `resultados_similaridade/`.

---

## 14) Resumo executivo

- A metodologia atual combina tecnicas de NLP (TF-IDF + cosseno), similaridade de conjuntos (Jaccard) e comparacao categorica.
- A nota final e transparente, auditavel e ponderada por dimensoes alinhadas ao contexto de P,D&I.
- O modelo permite rastrear por que um par recebeu determinada nota, facilitando auditoria e tomada de decisao.
