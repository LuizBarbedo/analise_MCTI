"""
Motor RAG (Retrieval-Augmented Generation)
Indexa e busca contexto relevante da base de conhecimento
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# TODO: Descomentar quando instalar dependências
# import chromadb
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer


@dataclass
class DocumentoRecuperado:
    """Documento recuperado pela busca"""
    conteudo: str
    fonte: str
    score: float
    metadata: dict


class RAGEngine:
    """
    Motor de busca semântica para base de conhecimento

    Usa ChromaDB para armazenar embeddings e realizar buscas por similaridade.
    Os documentos são chunked e indexados para recuperação eficiente.
    """

    def __init__(
        self,
        base_path: Path,
        modelo_embedding: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "lei_do_bem"
    ):
        """
        Inicializa o motor RAG

        Args:
            base_path: Caminho para pasta com documentos base
            modelo_embedding: Modelo para gerar embeddings
            collection_name: Nome da collection no ChromaDB
        """
        self.base_path = Path(base_path)
        self.modelo_embedding = modelo_embedding
        self.collection_name = collection_name

        # TODO: Inicializar quando dependências estiverem instaladas
        self._client = None
        self._collection = None
        self._embedder = None

    def _inicializar_chromadb(self):
        """Inicializa ChromaDB e collection"""
        # TODO: Implementar
        # self._client = chromadb.PersistentClient(path=str(self.base_path.parent / "embeddings"))
        # embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name=self.modelo_embedding
        # )
        # self._collection = self._client.get_or_create_collection(
        #     name=self.collection_name,
        #     embedding_function=embedding_fn
        # )
        pass

    def indexar_documentos(self, reindexar: bool = False) -> int:
        """
        Indexa todos os documentos da base de conhecimento

        Args:
            reindexar: Se True, reindexar mesmo se já existir

        Returns:
            Número de chunks indexados
        """
        if self._collection is None:
            self._inicializar_chromadb()

        documentos = []
        ids = []
        metadatas = []

        # Processa cada arquivo da base
        for arquivo in self.base_path.glob("*.txt"):
            chunks = self._chunkar_documento(arquivo)
            for i, chunk in enumerate(chunks):
                doc_id = f"{arquivo.stem}_{i}"
                documentos.append(chunk)
                ids.append(doc_id)
                metadatas.append({
                    "fonte": arquivo.name,
                    "chunk_index": i,
                    "tipo": self._identificar_tipo(arquivo.name)
                })

        # TODO: Adicionar ao ChromaDB
        # if documentos:
        #     self._collection.add(
        #         documents=documentos,
        #         ids=ids,
        #         metadatas=metadatas
        #     )

        return len(documentos)

    def _chunkar_documento(
        self,
        arquivo: Path,
        tamanho_chunk: int = 1000,
        overlap: int = 200
    ) -> list[str]:
        """
        Divide documento em chunks com overlap

        Args:
            arquivo: Caminho do arquivo
            tamanho_chunk: Tamanho alvo de cada chunk em caracteres
            overlap: Sobreposição entre chunks

        Returns:
            Lista de chunks
        """
        with open(arquivo, encoding="utf-8") as f:
            texto = f.read()

        chunks = []
        inicio = 0

        while inicio < len(texto):
            fim = inicio + tamanho_chunk

            # Tenta quebrar em fim de parágrafo
            if fim < len(texto):
                quebra = texto.rfind("\n\n", inicio, fim)
                if quebra > inicio:
                    fim = quebra

            chunk = texto[inicio:fim].strip()
            if chunk:
                chunks.append(chunk)

            inicio = fim - overlap if fim < len(texto) else len(texto)

        return chunks

    def _identificar_tipo(self, nome_arquivo: str) -> str:
        """Identifica o tipo de documento pelo nome"""
        nome_lower = nome_arquivo.lower()
        if "lei" in nome_lower or "11196" in nome_lower:
            return "legislacao"
        elif "manual" in nome_lower:
            return "manual"
        elif "frascati" in nome_lower:
            return "referencia_internacional"
        elif "decreto" in nome_lower:
            return "decreto"
        return "outro"

    def buscar(
        self,
        query: str,
        n_resultados: int = 5,
        filtro_tipo: Optional[str] = None
    ) -> list[DocumentoRecuperado]:
        """
        Busca documentos relevantes para a query

        Args:
            query: Texto de busca
            n_resultados: Número máximo de resultados
            filtro_tipo: Filtrar por tipo de documento

        Returns:
            Lista de DocumentoRecuperado ordenados por relevância
        """
        if self._collection is None:
            self._inicializar_chromadb()

        # TODO: Implementar busca real
        # where_filter = {"tipo": filtro_tipo} if filtro_tipo else None
        # results = self._collection.query(
        #     query_texts=[query],
        #     n_results=n_resultados,
        #     where=where_filter
        # )

        # Placeholder
        return [
            DocumentoRecuperado(
                conteudo="[Placeholder - Implementar busca RAG]",
                fonte="placeholder.txt",
                score=0.0,
                metadata={}
            )
        ]

    def buscar_contexto_avaliacao(self, projeto: dict) -> str:
        """
        Busca contexto relevante para avaliar um projeto específico

        Args:
            projeto: Dicionário com dados do projeto

        Returns:
            Contexto concatenado para usar no prompt
        """
        # Monta query combinando campos do projeto
        queries = [
            projeto.get("elemento_novo", ""),
            projeto.get("barreira_tecnologica", ""),
            projeto.get("metodologia", "")
        ]

        todos_resultados = []
        for query in queries:
            if query.strip():
                resultados = self.buscar(query, n_resultados=3)
                todos_resultados.extend(resultados)

        # Remove duplicatas e ordena por score
        vistos = set()
        unicos = []
        for r in sorted(todos_resultados, key=lambda x: x.score, reverse=True):
            if r.conteudo not in vistos:
                vistos.add(r.conteudo)
                unicos.append(r)

        # Formata contexto
        contexto_parts = []
        for r in unicos[:10]:  # Top 10
            contexto_parts.append(f"[Fonte: {r.fonte}]\n{r.conteudo}")

        return "\n\n---\n\n".join(contexto_parts)


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path

    base_path = Path(__file__).parent.parent / "data" / "base_conhecimento"
    rag = RAGEngine(base_path)

    # Indexar (primeira vez)
    # n_chunks = rag.indexar_documentos()
    # print(f"Indexados {n_chunks} chunks")

    # Buscar
    resultados = rag.buscar("novidade tecnológica inovação")
    for r in resultados:
        print(f"[{r.score:.3f}] {r.fonte}: {r.conteudo[:100]}...")
