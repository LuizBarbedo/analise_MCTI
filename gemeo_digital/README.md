# Gêmeo Digital - Avaliador Lei do Bem (Eletroeletrônica)

## Visão Geral

Este sistema simula o comportamento de um avaliador técnico do MCTI para projetos
de P&D&I submetidos à Lei do Bem na área de **Eletroeletrônica**.

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GÊMEO DIGITAL - AVALIADOR                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  BASE DE        │    │  CRITÉRIOS      │    │  EXEMPLOS       │         │
│  │  CONHECIMENTO   │    │  DE AVALIAÇÃO   │    │  FEW-SHOT       │         │
│  │  (RAG)          │    │  (Regras)       │    │  (Aprendizado)  │         │
│  │                 │    │                 │    │                 │         │
│  │ • Lei 11.196    │    │ • 4 Pilares     │    │ • Aprovados     │         │
│  │ • Manual 2025   │    │ • Frascati      │    │ • Rejeitados    │         │
│  │ • Frascati      │    │ • Red/Green     │    │ • Limítrofes    │         │
│  │ • Decreto 5798  │    │   Flags         │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                              │
│                    │      LLM (Claude)       │                              │
│                    │   Prompt Engineering    │                              │
│                    │   + Context Window      │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│                                ▼                                            │
│                    ┌─────────────────────────┐                              │
│                    │   MÓDULO DE AVALIAÇÃO   │                              │
│                    │                         │                              │
│                    │  • Análise por Pilar    │                              │
│                    │  • Scoring (0-10)       │                              │
│                    │  • Parecer Técnico      │                              │
│                    │  • Recomendações        │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│                                ▼                                            │
│                    ┌─────────────────────────┐                              │
│                    │       OUTPUT            │                              │
│                    │                         │                              │
│                    │  • Probabilidade        │                              │
│                    │    Aprovação (%)        │                              │
│                    │  • Parecer Detalhado    │                              │
│                    │  • Pontos a Melhorar    │                              │
│                    │  • Sugestões de Texto   │                              │
│                    └─────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Estratégia de "Treinamento"

### Por que NÃO treinar uma LLM do zero?

1. **Custo Proibitivo**: Fine-tuning de LLMs grandes custa milhares de dólares
2. **Dados Insuficientes**: ~60 projetos não são suficientes para fine-tuning robusto
3. **Manutenção**: Legislação muda, precisaríamos retreinar constantemente

### Abordagem Escolhida: RAG + Few-Shot Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABORDAGEM HÍBRIDA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. RAG (Retrieval-Augmented Generation)                        │
│     ├── Indexa toda documentação legal/técnica                  │
│     ├── Busca contexto relevante para cada avaliação            │
│     └── Garante aderência à legislação vigente                  │
│                                                                 │
│  2. Few-Shot Learning (Exemplos no Prompt)                      │
│     ├── Projetos aprovados → padrões a seguir                   │
│     ├── Projetos rejeitados → padrões a evitar                  │
│     └── Casos limítrofes → threshold de decisão                 │
│                                                                 │
│  3. Prompt Engineering Estruturado                              │
│     ├── System prompt com persona de avaliador MCTI             │
│     ├── Critérios explícitos com pesos                          │
│     └── Formato de saída padronizado                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## LLM Utilizada

### Opções e Trade-offs

| Modelo | Prós | Contras | Custo |
|--------|------|---------|-------|
| **Claude 3.5 Sonnet** | Melhor raciocínio, contexto 200k | Custo médio | ~$3/1M tokens |
| **GPT-4 Turbo** | Robusto, bem documentado | Contexto menor (128k) | ~$10/1M tokens |
| **Claude 3 Haiku** | Muito barato, rápido | Menos sofisticado | ~$0.25/1M tokens |
| **Llama 3 70B** | Gratuito, local | Requer GPU, menos capaz | Hardware |

### Recomendação: **Claude 3.5 Sonnet** (via API Anthropic)

- Contexto de 200k tokens permite carregar toda documentação + exemplos
- Raciocínio superior para análise técnica complexa
- Bom custo-benefício para uso em produção

## Componentes do Sistema

```
gemeo_digital/
├── README.md                    # Este arquivo
├── config/
│   ├── criterios_avaliacao.yaml # Critérios e pesos
│   └── prompts.yaml             # Templates de prompts
├── data/
│   ├── base_conhecimento/       # Documentação indexada
│   │   ├── lei_11196.txt
│   │   ├── manual_2025.txt
│   │   └── frascati.txt
│   ├── exemplos/
│   │   ├── aprovados/           # Projetos aprovados
│   │   ├── rejeitados/          # Projetos rejeitados
│   │   └── limitrofes/          # Rejeitados→Aprovados
│   └── embeddings/              # Vetores para RAG
├── src/
│   ├── __init__.py
│   ├── avaliador.py             # Classe principal
│   ├── rag_engine.py            # Motor de busca semântica
│   ├── prompt_builder.py        # Construtor de prompts
│   ├── scorer.py                # Sistema de pontuação
│   └── report_generator.py      # Gerador de pareceres
├── api/
│   └── main.py                  # FastAPI para servir
└── tests/
    └── test_avaliador.py
```

## Fluxo de Avaliação

```
PROJETO SUBMETIDO
       │
       ▼
┌──────────────────┐
│ 1. PRÉ-PROCESSAMENTO
│    • Extrai campos do projeto
│    • Valida formato
│    • Identifica área (eletroeletrônica)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. BUSCA RAG
│    • Encontra trechos relevantes da legislação
│    • Recupera projetos similares (aprovados/rejeitados)
│    • Monta contexto
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. AVALIAÇÃO POR PILAR
│    • Novidade Tecnológica (0-10)
│    • Barreira Técnica (0-10)
│    • Metodologia (0-10)
│    • Descrição (0-10)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. ANÁLISE FRASCATI
│    • Inovadora? (S/N)
│    • Criativa? (S/N)
│    • Incerta? (S/N)
│    • Sistemática? (S/N)
│    • Transferível? (S/N)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. SCORING FINAL
│    • Calcula score ponderado
│    • Compara com threshold
│    • Define: APROVADO / DILIGÊNCIA / REJEITADO
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. GERAÇÃO DE PARECER
│    • Justificativa por pilar
│    • Pontos fortes
│    • Pontos a melhorar
│    • Sugestões de reescrita
└──────────────────┘
```

## Sistema de Scoring

```yaml
# Pesos por Pilar
pesos:
  novidade_tecnologica: 0.30   # 30%
  barreira_tecnica: 0.30       # 30%
  metodologia: 0.25            # 25%
  descricao: 0.15              # 15%

# Thresholds de Decisão
thresholds:
  aprovado: >= 7.0             # Score >= 7 → APROVADO
  diligencia: >= 5.0           # 5 <= Score < 7 → DILIGÊNCIA
  rejeitado: < 5.0             # Score < 5 → REJEITADO

# Critérios Frascati (obrigatórios)
frascati:
  minimo_criterios: 4          # Mínimo 4 de 5 critérios
  criterios_obrigatorios:
    - inovadora
    - incerta
```

## Uso do Sistema

```python
from gemeo_digital import AvaliadorLeiBem

# Inicializa o avaliador
avaliador = AvaliadorLeiBem(
    area="eletroeletronica",
    modelo="claude-3-5-sonnet-20241022"
)

# Carrega projeto a ser avaliado
projeto = {
    "titulo": "Sistema de Monitoramento IoT para Redes Elétricas",
    "descricao": "...",
    "elemento_novo": "...",
    "barreira_tecnologica": "...",
    "metodologia": "..."
}

# Avalia
resultado = avaliador.avaliar(projeto)

# Resultado
print(f"Score: {resultado.score}/10")
print(f"Decisão: {resultado.decisao}")
print(f"Probabilidade de Aprovação: {resultado.prob_aprovacao}%")
print(f"Parecer: {resultado.parecer}")
```

## Próximos Passos

1. [ ] Receber bases de projetos (aprovados/rejeitados/limítrofes)
2. [ ] Indexar base de conhecimento (RAG)
3. [ ] Calibrar thresholds com dados reais
4. [ ] Testar e validar com exemplos conhecidos
5. [ ] Ajustar prompts baseado em feedback
6. [ ] Deploy via API

## Requisitos

```
python >= 3.10
anthropic >= 0.18.0
chromadb >= 0.4.0
sentence-transformers >= 2.2.0
fastapi >= 0.100.0
pydantic >= 2.0.0
pyyaml >= 6.0
```
