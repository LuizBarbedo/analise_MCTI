# Gêmeo Digital - Avaliador Lei do Bem
## Documento Técnico de Arquitetura e Funcionamento

**Versão:** 1.0
**Data:** Março 2025
**Área:** Eletroeletrônica

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Por que Claude 3.5 Sonnet?](#3-por-que-claude-35-sonnet)
4. [Fluxo de Avaliação](#4-fluxo-de-avaliação)
5. [Sistema de Scoring](#5-sistema-de-scoring)
6. [Estrutura de Arquivos](#6-estrutura-de-arquivos)
7. [Próximos Passos](#7-próximos-passos)

---

## 1. Visão Geral

O **Gêmeo Digital** é um sistema de inteligência artificial que simula o comportamento
de um avaliador técnico do MCTI para projetos de P&D&I submetidos à Lei do Bem
(Lei 11.196/2005).

### Objetivo

Permitir que empresas **pré-avaliem** seus projetos antes de submetê-los oficialmente,
identificando:

- Pontos fortes que devem ser mantidos
- Pontos fracos que precisam de melhoria
- Probabilidade de aprovação
- Sugestões específicas de reescrita

### Escopo Atual

| Área | Status |
|------|--------|
| Eletroeletrônica | Em desenvolvimento |
| Química | Planejado |
| Software/TIC | Planejado |
| Mecânica | Planejado |
| Biotecnologia | Planejado |

---

## 2. Arquitetura do Sistema

### 2.1 Abordagem: RAG + Few-Shot Learning

**Importante:** NÃO treinamos uma LLM do zero. Isso seria:
- Extremamente caro (milhares de dólares)
- Inviável com poucos exemplos (~60 projetos)
- Difícil de manter quando a legislação muda

Em vez disso, usamos uma **abordagem híbrida** que combina três técnicas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARQUITETURA HÍBRIDA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐                                                   │
│   │  1. RAG             │  Retrieval-Augmented Generation                   │
│   │  (Recuperação)      │                                                   │
│   │                     │  • Indexa toda documentação legal                 │
│   │  ChromaDB +         │  • Busca trechos relevantes para cada projeto     │
│   │  Embeddings         │  • Garante aderência à legislação vigente         │
│   │                     │  • Atualização simples: troca o documento         │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  2. FEW-SHOT        │  Aprendizado por Exemplos                         │
│   │  LEARNING           │                                                   │
│   │                     │  • Projetos APROVADOS → padrões de excelência     │
│   │  Exemplos no        │  • Projetos REJEITADOS → padrões a evitar         │
│   │  Contexto           │  • Projetos LIMÍTROFES → calibração do threshold  │
│   │                     │  • LLM aprende "no ato" com os exemplos           │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  3. PROMPT          │  Engenharia de Prompts                            │
│   │  ENGINEERING        │                                                   │
│   │                     │  • System prompt com persona de avaliador MCTI    │
│   │  Instruções         │  • Critérios explícitos com pesos definidos       │
│   │  Estruturadas       │  • Formato de saída padronizado                   │
│   │                     │  • Checklist de red/green flags                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │                      CLAUDE 3.5 SONNET                              │   │
│   │                   (200k tokens de contexto)                         │   │
│   │                                                                     │   │
│   │    Recebe:                        Retorna:                          │   │
│   │    • Contexto legal (RAG)         • Scores por pilar (0-10)         │   │
│   │    • Exemplos (Few-shot)          • Análise Frascati (5 critérios)  │   │
│   │    • Instruções (Prompt)          • Decisão final                   │   │
│   │    • Projeto a avaliar            • Parecer detalhado               │   │
│   │                                   • Sugestões de melhoria           │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Componentes Detalhados

#### 2.2.1 RAG (Retrieval-Augmented Generation)

O RAG permite que o sistema "consulte" a legislação em tempo real:

```
Base de Conhecimento Indexada:
├── Lei 11.196/2005 (Lei do Bem)
├── Decreto 5.798/2006
├── Manual Lei do Bem 2025
├── Manual de Frascati (OCDE)
└── Portarias e INs relevantes

Processo:
1. Projeto é submetido
2. Sistema extrai palavras-chave e conceitos
3. Busca vetorial encontra trechos mais relevantes
4. Contexto é injetado no prompt da LLM
```

**Tecnologias:**
- **ChromaDB**: Banco de dados vetorial para armazenar embeddings
- **Sentence-Transformers**: Modelo para gerar embeddings multilíngues
- **Chunking**: Documentos divididos em pedaços de ~1000 caracteres com overlap

#### 2.2.2 Few-Shot Learning

O sistema aprende padrões diretamente dos exemplos:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXEMPLOS NO CONTEXTO                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  APROVADOS (20 projetos)                                        │
│  ├── Elemento novo bem descrito                                 │
│  ├── Barreira técnica clara                                     │
│  ├── Metodologia experimental                                   │
│  └── Resultados mensuráveis                                     │
│                                                                 │
│  REJEITADOS (20 projetos)                                       │
│  ├── Descrição genérica                                         │
│  ├── Sem barreira técnica real                                  │
│  ├── Metodologia de gestão apenas                               │
│  └── Confunde rotina com P&D                                    │
│                                                                 │
│  LIMÍTROFES (20 projetos)                                       │
│  ├── Rejeitados que foram aprovados após ajustes                │
│  ├── Mostram o "threshold" de decisão                           │
│  └── Indicam o mínimo aceitável                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 Prompt Engineering

O prompt é estruturado em camadas:

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT (Persona)                                         │
├─────────────────────────────────────────────────────────────────┤
│ "Você é um avaliador técnico especialista do MCTI,              │
│  responsável por analisar projetos de P&D&I..."                 │
│                                                                 │
│  • Princípios de avaliação                                      │
│  • O que verificar                                              │
│  • O que rejeitar                                               │
│  • Formato de resposta                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CONTEXTO RAG                                                    │
├─────────────────────────────────────────────────────────────────┤
│ [Trechos relevantes da legislação encontrados pela busca]       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXEMPLOS FEW-SHOT                                               │
├─────────────────────────────────────────────────────────────────┤
│ "Projetos APROVADOS (padrão a seguir):"                         │
│ [Exemplos de projetos aprovados]                                │
│                                                                 │
│ "Projetos REJEITADOS (padrões a evitar):"                       │
│ [Exemplos de projetos rejeitados]                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PROJETO A AVALIAR                                               │
├─────────────────────────────────────────────────────────────────┤
│ Título: [...]                                                   │
│ Descrição: [...]                                                │
│ Elemento Novo: [...]                                            │
│ Barreira Tecnológica: [...]                                     │
│ Metodologia: [...]                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ INSTRUÇÕES DE AVALIAÇÃO                                         │
├─────────────────────────────────────────────────────────────────┤
│ "Avalie o projeto seguindo EXATAMENTE esta estrutura:           │
│  1. Avaliação por Pilar (0-10)                                  │
│  2. Análise Frascati (5 critérios)                              │
│  3. Score Final e Decisão                                       │
│  4. Parecer Técnico                                             │
│  5. Sugestões de Melhoria"                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Por que Claude 3.5 Sonnet?

### 3.1 Comparativo de Modelos

| Modelo | Contexto | Raciocínio | Português | Custo/1M tokens |
|--------|----------|------------|-----------|-----------------|
| **Claude 3.5 Sonnet** | **200k** | **Excelente** | **Excelente** | **~$3** |
| GPT-4 Turbo | 128k | Excelente | Muito bom | ~$10 |
| GPT-4o | 128k | Muito bom | Muito bom | ~$5 |
| Claude 3 Haiku | 200k | Bom | Bom | ~$0.25 |
| Llama 3 70B | 8k | Bom | Razoável | GPU local |

### 3.2 Motivos da Escolha

#### 3.2.1 Contexto de 200k Tokens

```
O que cabe em 200k tokens:
├── Lei do Bem completa (~15k tokens)
├── Manual 2025 (~40k tokens)
├── Manual Frascati (trechos, ~20k tokens)
├── 10 projetos aprovados (~30k tokens)
├── 10 projetos rejeitados (~30k tokens)
├── Projeto sendo avaliado (~3k tokens)
├── Instruções e prompts (~5k tokens)
└── SOBRA: ~57k tokens para resposta detalhada
```

Isso permite avaliar **com toda informação necessária numa única chamada**.

#### 3.2.2 Raciocínio Superior

Claude 3.5 Sonnet demonstra capacidade superior para:
- Análise técnica complexa
- Comparação entre documentos
- Identificação de inconsistências
- Geração de pareceres estruturados

#### 3.2.3 Custo-Benefício

Para avaliação de 100 projetos/mês:
- **Claude 3.5**: ~$30-50/mês
- **GPT-4 Turbo**: ~$100-150/mês
- **Fine-tuning próprio**: $5.000+ inicial + infraestrutura

#### 3.2.4 Português Excelente

Crucial porque:
- Documentação legal está em português
- Projetos são escritos em português
- Pareceres precisam ser em português técnico adequado

---

## 4. Fluxo de Avaliação

### 4.1 Diagrama do Fluxo Completo

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FLUXO DE AVALIAÇÃO                                 │
└──────────────────────────────────────────────────────────────────────────────┘

    ENTRADA                                                          SAÍDA
       │                                                               │
       ▼                                                               ▼
┌─────────────┐                                                 ┌─────────────┐
│   PROJETO   │                                                 │  RESULTADO  │
│             │                                                 │             │
│ • Título    │                                                 │ • Score     │
│ • Descrição │                                                 │ • Decisão   │
│ • Elem.Novo │                                                 │ • Parecer   │
│ • Barreira  │                                                 │ • Sugestões │
│ • Metodol.  │                                                 │             │
└──────┬──────┘                                                 └──────▲──────┘
       │                                                               │
       │  ┌────────────────────────────────────────────────────────────┘
       │  │
       ▼  │
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 1: PRÉ-PROCESSAMENTO                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • Extrai campos do projeto                                                 │
│  • Valida formato e completude                                              │
│  • Identifica área (eletroeletrônica)                                       │
│  • Normaliza texto (remove caracteres especiais)                            │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 2: BUSCA RAG                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: elemento_novo + barreira + metodologia                              │
│                     │                                                       │
│                     ▼                                                       │
│            ┌─────────────────┐                                              │
│            │    ChromaDB     │                                              │
│            │   (Embeddings)  │                                              │
│            └────────┬────────┘                                              │
│                     │                                                       │
│                     ▼                                                       │
│  Resultados:                                                                │
│  • Top 10 trechos mais relevantes da legislação                             │
│  • Score de similaridade para cada trecho                                   │
│  • Fonte identificada (Lei, Manual, Decreto)                                │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 3: RECUPERAÇÃO DE EXEMPLOS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Busca por similaridade semântica:                                          │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │    APROVADOS    │  │   REJEITADOS    │  │   LIMÍTROFES    │              │
│  │                 │  │                 │  │                 │              │
│  │  Top 3 mais     │  │  Top 3 mais     │  │  Top 2 mais     │              │
│  │  similares      │  │  similares      │  │  similares      │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 4: MONTAGEM DO PROMPT                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  prompt_final = f"""                                                        │
│      {system_prompt}           # Persona de avaliador MCTI                  │
│      {contexto_rag}            # Trechos da legislação                      │
│      {exemplos_aprovados}      # Padrões a seguir                           │
│      {exemplos_rejeitados}     # Padrões a evitar                           │
│      {projeto_avaliar}         # Projeto sendo avaliado                     │
│      {instrucoes_avaliacao}    # Estrutura de resposta                      │
│  """                                                                        │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 5: AVALIAÇÃO POR PILAR (Claude 3.5 Sonnet)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   PILAR 1: DESCRIÇÃO DO PROJETO (Peso: 15%)                         │    │
│  │   ├── Objetivo de inovação claro?                                   │    │
│  │   ├── Classificação correta (PB/PA/DE)?                             │    │
│  │   ├── TRL coerente?                                                 │    │
│  │   └── Score: X/10                                                   │    │
│  │                                                                     │    │
│  │   PILAR 2: ELEMENTO NOVO/INOVADOR (Peso: 30%)                       │    │
│  │   ├── Novidade tecnológica identificada?                            │    │
│  │   ├── Comparação com estado da arte?                                │    │
│  │   ├── Não-obviedade demonstrada?                                    │    │
│  │   └── Score: X/10                                                   │    │
│  │                                                                     │    │
│  │   PILAR 3: BARREIRA TECNOLÓGICA (Peso: 30%)                         │    │
│  │   ├── Incerteza técnica real?                                       │    │
│  │   ├── Problema não trivial?                                         │    │
│  │   ├── Abordagem de superação descrita?                              │    │
│  │   └── Score: X/10                                                   │    │
│  │                                                                     │    │
│  │   PILAR 4: METODOLOGIA (Peso: 25%)                                  │    │
│  │   ├── Protocolo experimental estruturado?                           │    │
│  │   ├── Hipóteses formuladas e testadas?                              │    │
│  │   ├── Resultados validados?                                         │    │
│  │   └── Score: X/10                                                   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 6: ANÁLISE FRASCATI                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Verifica os 5 critérios do Manual de Frascati (OCDE):                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  [ ] 1. INOVADORA (obrigatório)                                     │    │
│  │      "Visa resultados novos"                                        │    │
│  │                                                                     │    │
│  │  [ ] 2. CRIATIVA                                                    │    │
│  │      "Baseia-se em conceitos e hipóteses originais"                 │    │
│  │                                                                     │    │
│  │  [ ] 3. INCERTA (obrigatório)                                       │    │
│  │      "Apresenta incerteza quanto ao resultado final"                │    │
│  │                                                                     │    │
│  │  [ ] 4. SISTEMÁTICA                                                 │    │
│  │      "É planejada e orçamentada"                                    │    │
│  │                                                                     │    │
│  │  [ ] 5. TRANSFERÍVEL                                                │    │
│  │      "Conduz a resultados reprodutíveis"                            │    │
│  │                                                                     │    │
│  │  Mínimo para aprovação: 4/5 critérios                               │    │
│  │  Critérios obrigatórios: INOVADORA + INCERTA                        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 7: CÁLCULO DO SCORE FINAL                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Score = (Descrição × 0.15) + (Elem.Novo × 0.30) +                          │
│          (Barreira × 0.30) + (Metodologia × 0.25)                           │
│                                                                             │
│  Exemplo:                                                                   │
│  Score = (7.0 × 0.15) + (6.5 × 0.30) + (6.0 × 0.30) + (7.0 × 0.25)          │
│  Score = 1.05 + 1.95 + 1.80 + 1.75 = 6.55                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                      DECISÃO FINAL                              │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                 │        │
│  │  Score >= 7.0 + Frascati >= 4  →  APROVADO                      │        │
│  │  Score >= 5.0 e < 7.0          →  DILIGÊNCIA                    │        │
│  │  Score < 5.0                   →  REJEITADO                     │        │
│  │                                                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 8: GERAÇÃO DO PARECER                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT FINAL:                                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  RESULTADO DA AVALIAÇÃO                                             │    │
│  │  ═══════════════════════════════════════════════════════════════    │    │
│  │                                                                     │    │
│  │  Score Geral: 6.55/10                                               │    │
│  │  Decisão: DILIGÊNCIA                                                │    │
│  │  Probabilidade de Aprovação: 58%                                    │    │
│  │  Critérios Frascati: 4/5                                            │    │
│  │                                                                     │    │
│  │  PARECER RESUMIDO:                                                  │    │
│  │  O projeto apresenta potencial de inovação tecnológica,             │    │
│  │  porém necessita de ajustes na descrição da barreira                │    │
│  │  tecnológica e na metodologia experimental...                       │    │
│  │                                                                     │    │
│  │  PONTOS FORTES:                                                     │    │
│  │  • Elemento novo bem identificado                                   │    │
│  │  • TRL coerente com atividades                                      │    │
│  │                                                                     │    │
│  │  PONTOS A MELHORAR:                                                 │    │
│  │  • Barreira técnica precisa ser mais específica                     │    │
│  │  • Metodologia deve detalhar hipóteses testadas                     │    │
│  │                                                                     │    │
│  │  SUGESTÕES DE REESCRITA:                                            │    │
│  │  [Sugestões específicas para cada campo com problemas]              │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Tempo de Processamento Estimado

| Etapa | Tempo |
|-------|-------|
| Pré-processamento | ~100ms |
| Busca RAG | ~500ms |
| Recuperação exemplos | ~300ms |
| Montagem prompt | ~50ms |
| Chamada LLM | ~10-30s |
| Parsing resposta | ~100ms |
| **Total** | **~15-35 segundos** |

---

## 5. Sistema de Scoring

### 5.1 Pesos por Pilar

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUIÇÃO DE PESOS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ELEMENTO NOVO          ████████████████████████████░░  30%    │
│   (Novidade tecnológica)                                        │
│                                                                 │
│   BARREIRA TECNOLÓGICA   ████████████████████████████░░  30%    │
│   (Incerteza técnica)                                           │
│                                                                 │
│   METODOLOGIA            ██████████████████████░░░░░░░░  25%    │
│   (Método experimental)                                         │
│                                                                 │
│   DESCRIÇÃO              ████████████░░░░░░░░░░░░░░░░░░  15%    │
│   (Clareza e estrutura)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Critérios de Decisão

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATRIZ DE DECISÃO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  APROVADO                                                       │
│  ─────────                                                      │
│  • Score >= 7.0                                                 │
│  • Frascati >= 4/5                                              │
│  • Critérios INOVADORA e INCERTA atendidos                      │
│  • Nenhum pilar crítico < 6.0                                   │
│                                                                 │
│  DILIGÊNCIA                                                     │
│  ──────────                                                     │
│  • Score entre 5.0 e 6.99                                       │
│  OU                                                             │
│  • Score >= 7.0 mas Frascati = 3/5                              │
│  OU                                                             │
│  • Algum pilar crítico entre 5.0 e 5.99                         │
│                                                                 │
│  REJEITADO                                                      │
│  ─────────                                                      │
│  • Score < 5.0                                                  │
│  OU                                                             │
│  • Frascati < 3/5                                               │
│  OU                                                             │
│  • Critério INOVADORA ou INCERTA não atendido                   │
│  OU                                                             │
│  • Algum pilar crítico < 5.0                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Red Flags e Green Flags

```
┌─────────────────────────────────────────────────────────────────┐
│  RED FLAGS (Indicadores de Rejeição)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ Descrição genérica sem detalhes técnicos                    │
│  ❌ Confunde rotina operacional com P&D                         │
│  ❌ Foco apenas em benefícios comerciais                        │
│  ❌ Metodologia apenas de gestão (Scrum, PMBOK)                 │
│  ❌ Sem comparação com estado da arte                           │
│  ❌ Barreira apenas de prazo/custo/gestão                       │
│  ❌ Customização de produto existente                           │
│  ❌ "Otimizamos", "melhoramos" sem especificar como             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GREEN FLAGS (Indicadores de Aprovação)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ Algoritmo/arquitetura/protocolo próprio identificado        │
│  ✅ Comparação quantitativa com soluções existentes             │
│  ✅ Patente depositada ou em análise                            │
│  ✅ Hipóteses técnicas testadas e documentadas                  │
│  ✅ Descrição de falhas e iterações                             │
│  ✅ Métricas técnicas de sucesso definidas                      │
│  ✅ Ensaios laboratoriais documentados                          │
│  ✅ Prototipagem com iterações                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Estrutura de Arquivos

```
gemeo_digital/
│
├── README.md                           # Documentação principal
├── COMO_FUNCIONA.md                    # Este documento
├── requirements.txt                    # Dependências Python
│
├── config/
│   ├── criterios_avaliacao.yaml       # Critérios, pesos, thresholds
│   └── prompts.yaml                    # Templates de prompts
│
├── data/
│   ├── base_conhecimento/             # Documentos para RAG
│   │   ├── lei do bem 11196_2005.txt
│   │   └── manual_lei_bem_2025.txt
│   │
│   ├── exemplos/                       # Projetos para few-shot
│   │   ├── aprovados/                  # [A popular com dados reais]
│   │   ├── rejeitados/                 # [A popular com dados reais]
│   │   └── limitrofes/                 # [A popular com dados reais]
│   │
│   └── embeddings/                     # Vetores ChromaDB
│
├── src/
│   ├── __init__.py
│   ├── avaliador.py                    # Classe principal
│   └── rag_engine.py                   # Motor de busca semântica
│
├── api/
│   └── main.py                         # API FastAPI
│
└── tests/
    └── test_avaliador.py               # Testes automatizados
```

---

## 7. Próximos Passos

### 7.1 Itens Pendentes

| # | Item | Status | Dependência |
|---|------|--------|-------------|
| 1 | Base de projetos aprovados | Aguardando | Dados do usuário |
| 2 | Base de projetos rejeitados | Aguardando | Dados do usuário |
| 3 | Base de projetos limítrofes | Aguardando | Dados do usuário |
| 4 | Integração API Claude | Pronto para implementar | API key |
| 5 | Indexação RAG | Pronto para executar | ChromaDB instalado |
| 6 | Calibração de thresholds | Após ter bases | Itens 1-3 |
| 7 | Testes de validação | Após calibração | Item 6 |
| 8 | Deploy API | Final | Todos anteriores |

### 7.2 Formato Esperado das Bases

Para alimentar o sistema, os projetos devem estar no formato:

```yaml
# Exemplo: aprovados/projeto_001.yaml
titulo: "Sistema de Monitoramento IoT..."
descricao: |
  Desenvolvimento de um sistema embarcado...
elemento_novo: |
  Algoritmo proprietário de detecção...
barreira_tecnologica: |
  Desafio de executar modelos de ML...
metodologia: |
  1. Revisão do estado da arte
  2. Prototipagem...
resultado: "APROVADO"
ano_base: 2023
observacoes: "Aprovado de primeira, sem diligências"
```

---

**Documento gerado em:** Março 2025
**Sistema:** Gêmeo Digital - Avaliador Lei do Bem v0.1.0
**Área:** Eletroeletrônica
