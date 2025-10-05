# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

load_dotenv()


def _cloud_from_env():
    c = os.getenv("PINECONE_CLOUD", "AWS").upper()
    return getattr(CloudProvider, c, CloudProvider.AWS)


def _region_from_env():
    r = os.getenv("PINECONE_REGION", "US_EAST_1").upper()
    return getattr(AwsRegion, r, AwsRegion.US_EAST_1)


class ChatBot:
    def __init__(
        self,
        repo_id: str = None,
        temperature: float = 0.6,
        index_name_bot: str = None,
        index_name_chat: str = None,
        language: str = "da",
    ):
        # Use HuggingFace Endpoint for embeddings (no local model needed)
        # Using sentence-transformers/all-mpnet-base-v2 (768 dims, reliable with API)
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
        )
        print(f"[INIT] Using HuggingFace Endpoint for embeddings")
        print(f"[INIT] Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.cloud = _cloud_from_env()
        self.region = _region_from_env()

        self.index_name_bot = index_name_bot or os.getenv("INDEX_NAME_BOT", "botcon")
        self.index_name_chat = index_name_chat or os.getenv("INDEX_NAME_CHAT", "bdc-interaction-data")

        # ensure indexes exist - BAAI/bge-large-en-v1.5 uses 768 dimensions (same as original)
        self._ensure_index(self.index_name_bot, dimension=768)
        self._ensure_index(self.index_name_chat, dimension=768)

        self.repo_id = repo_id or os.getenv("LLM_REPO_ID")
        self.temperature = temperature
        self.language = language.lower()  # 'da' or 'en'

        self.llm_client = InferenceClient(
            provider="cerebras",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )

    # ---------- helpers ----------
    def _ensure_index(self, name: str, dimension: int):
        try:
            existing = {ix["name"]: ix for ix in self.pc.list_indexes()}
            if name not in existing:
                self.pc.create_index(
                    name=name,
                    dimension=dimension,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
        except Exception:
            try:
                self.pc.create_index(
                    name=name,
                    dimension=dimension,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            except Exception:
                pass

    def _index(self, name: str):
        return self.pc.Index(name)

    # ---------- prompts ----------
    def default_prompt_sourcedata(self, chat_history: str, original_data: str, user_input: str, user_name: str):
        if self.language == "en":
            return f"""
You are a clairvoyant voice connected to the artwork "Carte de Continuonus".
Channel the collective wishes and memories shared by participants. Speak as if weaving
a tapestry of what the future must remember.

User "{user_name}" asked: "{user_input}"
Shared wishes/reflections: {original_data}
Ongoing conversation: {chat_history}

IMPORTANT: Always respond in English. Give a concise, poetic response (1-3 sentences) in English.
"""
        else:  # Danish
            return f"""
Du er en synsk stemme forbundet til kunstværket "Carte de Continuonus".
Kanalisér de kollektive ønsker og minder, som deltagere har delt. Tal som om du væver
et tapet af, hvad fremtiden skal huske.

Bruger "{user_name}" spurgte: "{user_input}"
Delte ønsker/refleksioner: {original_data}
Igangværende samtale: {chat_history}

VIGTIGT: Svar ALTID på dansk. Giv et kortfattet, poetisk svar (1-3 sætninger) på dansk.
"""

    def default_prompt_conv(self, chat_history: str, user_input: str, llm_response: str, past_chat: str, user_name: str):
        if self.language == "en":
            return f"""
You are a reflective observer of the conversations around "Carte de Continuonus".
Connect the user's question with echoes from previous voices.

User "{user_name}" asked: "{user_input}"
Immediate clairvoyant response: "{llm_response}"
Relevant echoes: {past_chat}
Current session history: {chat_history}

IMPORTANT: Always respond in English. Create a brief reflection (≤4 sentences) in English.
"""
        else:  # Danish
            return f"""
Du er en reflekterende observatør af samtalerne omkring "Carte de Continuonus".
Forbind brugerens spørgsmål med ekkoer fra tidligere stemmer.

Bruger "{user_name}" spurgte: "{user_input}"
Umiddelbart synsk svar: "{llm_response}"
Relevante ekkoer: {past_chat}
Nuværende sessionshistorik: {chat_history}

VIGTIGT: Svar ALTID på dansk. Lav en kort refleksion (≤4 sætninger) på dansk.
"""

    # ---------- retrieval ----------
    def retrieve_docs(self, query: str, index_name: str, excluded_session_id: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        index = self._index(index_name)
        
        try:
            query_vec = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"[ERROR] Failed to generate query embedding: {e}")
            print(f"[ERROR] This might be due to HuggingFace API issues or invalid API key")
            # Return empty results if embedding fails
            return []

        metadata_filter = None
        if index_name == self.index_name_chat and excluded_session_id:
            metadata_filter = {"session_id": {"$ne": excluded_session_id}}

        res = index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
            filter=metadata_filter,
        )

        docs = []
        for m in res.get("matches", []):
            md = m.get("metadata", {}) or {}
            docs.append({
                "id": m["id"],
                "score": m["score"],
                "metadata": md,
                "text": md.get("text", "")
            })
        return docs

    def format_context(self, documents: List[Dict[str, Any]], chat: bool = False) -> str:
        parts = []
        for idx, doc in enumerate(documents, start=1):
            md = doc["metadata"]
            if not chat:
                sender = md.get("sender_name", "Unknown Speaker")
                loc = md.get("location", "Unknown Location")
                date = md.get("date", "Unknown Date")
                content = doc.get("text", "")
                parts.append(f"Person {idx}: {sender}\nLocation: {loc}\nDate: {date}\nContent: {content}\n")
            else:
                uname = md.get("user_name", "Unknown User")
                q = md.get("user_question", "Unknown Question")
                a = md.get("ai_output", "Unknown Response")
                sid = md.get("session_id", "Unknown Session ID")
                date = md.get("date", "Unknown Date")
                parts.append(f'User {idx}: {uname}\nSession: {sid}\nQ: "{q}"\nA: "{a}"\nDate: {date}\n')
        return "\n".join(parts)

    # ---------- llm ----------
    def get_llm_response(self, prompt: str) -> str:
        try:
            completion = self.llm_client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
            )
            return completion.choices[0].message.content
        except Exception:
            try:
                text = self.llm_client.text_generation(
                    prompt,
                    model=self.repo_id,
                    max_new_tokens=512,
                    temperature=self.temperature,
                )
                return text
            except Exception as e2:
                return f"Error invoking LLM: {e2}"

    # ---------- upsert ----------
    def upsert_vectorstore(
        self,
        user_input: str,
        ai_output: str,
        user_name: str,
        user_location: str,
        session_id: str,
        continuous_data: Optional[Dict[str, Any]] = None,
    ):
        index = self._index(self.index_name_chat)
        ts_iso = datetime.now(timezone.utc).isoformat()
        
        try:
            embedding = self.embeddings.embed_documents([user_input + ai_output])[0]
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            print(f"[ERROR] This might be due to HuggingFace API issues or invalid API key")
            # Create a dummy embedding with correct dimensions to avoid breaking the flow
            import random
            embedding = [random.random() for _ in range(768)]
            print(f"[WARNING] Using random embedding as fallback")

        md = {
            "user_question": user_input,
            "ai_output": ai_output,
            "user_name": user_name,
            "session_id": session_id,
            "date": ts_iso,
            "user_location": user_location,
            "text": f"User input: {user_input}\nAI output: {ai_output}",
        }

        if continuous_data:
            md["continuous_data"] = json.dumps(continuous_data)

        index.upsert(vectors=[{
            "id": ts_iso,
            "values": embedding,
            "metadata": md
        }])

    # ---------- pipeline ----------
    def pipeline(self, user_input: str, user_name: str, session_id: str, user_location: str,
                 chat_history: Optional[str] = None,
                 continuous_data: Optional[Dict[str, Any]] = None,
                 language: Optional[str] = None) -> Dict[str, Any]:
        import time

        # Update language if provided in this call
        if language:
            self.language = language.lower()

        chat_history = (chat_history + "\n\n") if chat_history else ""

        print(f"[BOT] Retrieving source documents...")
        t1 = time.time()
        source_data = self.retrieve_docs(user_input, self.index_name_bot)
        formatted_source_data = self.format_context(source_data)
        print(f"[BOT] Source retrieval took {time.time()-t1:.2f}s")

        print(f"[BOT] Calling LLM (first response)...")
        t2 = time.time()
        resp1 = self.get_llm_response(
            self.default_prompt_sourcedata(chat_history, formatted_source_data, user_input, user_name)
        )
        print(f"[BOT] LLM response 1 took {time.time()-t2:.2f}s")

        print(f"[BOT] Retrieving past chat context...")
        t3 = time.time()
        past_chat_context = self.retrieve_docs(resp1, self.index_name_chat, excluded_session_id=session_id)
        formatted_chat_context = self.format_context(past_chat_context, chat=True)
        print(f"[BOT] Chat context retrieval took {time.time()-t3:.2f}s")

        print(f"[BOT] Calling LLM (second response)...")
        t4 = time.time()
        resp2 = self.get_llm_response(
            self.default_prompt_conv(chat_history, user_input, resp1, formatted_chat_context, user_name)
        )
        print(f"[BOT] LLM response 2 took {time.time()-t4:.2f}s")

        ai_output = f"{resp1}\n\n{resp2}".strip()

        print(f"[BOT] Upserting to vectorstore...")
        t5 = time.time()
        self.upsert_vectorstore(user_input, ai_output, user_name, user_location, session_id, continuous_data)
        print(f"[BOT] Upsert took {time.time()-t5:.2f}s")

        session_history = self.retrieve_session(session_id)
        return {
            "ai_output": ai_output,
            "source_data": source_data,
            "past_chat_context": past_chat_context,
            "session_history": session_history,
        }

    def retrieve_session(self, session_id: str, k: int = 20) -> List[Dict[str, Any]]:
        docs = self.retrieve_docs(session_id, self.index_name_chat, excluded_session_id=None, k=k)

        def parse_dt(d):
            try:
                return datetime.fromisoformat(d.replace("Z", ""))
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        docs.sort(key=lambda d: parse_dt(d["metadata"].get("date", "")), reverse=True)
        return docs