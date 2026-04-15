"""RAG Agent — retrieval-augmented generation from ingested documents."""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Answer the user's question using "
            "ONLY the provided context. If the context does not contain the "
            "answer, say so honestly. Always cite the source documents.\n\n"
            "## Retrieved Context\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@register_agent
class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent.

    Searches the Qdrant vector store for relevant document chunks,
    then generates an answer grounded in those sources.
    """

    name = "rag"
    description = (
        "Retrieval-Augmented Generation. Searches uploaded documents "
        "and research papers to answer questions with citations. "
        "Use when the user asks about content in their documents."
    )

    async def run(self, state: AgentState) -> AgentState:
        llm = get_llm()

        # Get the latest user message
        messages = state.get("messages", [])
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
            elif hasattr(msg, "content") and not hasattr(msg, "type"):
                question = msg.content
                break

        if not question:
            question = str(messages[-1].content) if messages else ""

        # Retrieve context from RAG pipeline
        context_chunks = state.get("context", [])

        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[Source: {c.get('source', 'unknown')}]\n{c.get('content', '')}"
                for c in context_chunks
            )
            sources = [
                {"source": c.get("source", "unknown"), "score": c.get("score", 0)}
                for c in context_chunks
            ]
        else:
            # If no context was pre-loaded, try retriever
            try:
                from app.rag.retriever import retrieve_documents

                meta = state.get("metadata", {})
                doc_ids = meta.get("document_ids") or None
                user_id = meta.get("user_id") or None
                chat_id = meta.get("chat_id") or None
                results = await retrieve_documents(
                    question,
                    document_ids=doc_ids,
                    user_id=user_id,
                    chat_id=chat_id,
                )
                context_text = "\n\n---\n\n".join(
                    f"[Source: {r.get('source', 'unknown')}]\n{r.get('content', '')}"
                    for r in results
                )
                sources = [
                    {"source": r.get("source", "unknown"), "score": r.get("score", 0)}
                    for r in results
                ]
            except Exception:
                context_text = "(No documents found. Answer from general knowledge.)"
                sources = []

        chain = RAG_PROMPT | llm
        response = await chain.ainvoke(
            {"context": context_text, "question": question}
        )
        content = response.content if isinstance(response.content, str) else str(response.content)

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": sources,
            "metadata": {},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
