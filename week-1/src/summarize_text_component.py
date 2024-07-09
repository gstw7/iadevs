"""
This module provides the SummarizerChainComponent class for summarizing text.
"""
from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langflow.base.chains.model import LCChainComponent
from langflow.inputs import HandleInput, MessageTextInput


class SummarizerChainComponent(LCChainComponent):
    """Chain component for summarizing text."""
    display_name = "SummarizerChain"
    description = "Chain for summarizing text."
    name = "SummarizerChain"
    icon = "custom_components"

    inputs = [
        MessageTextInput(
            name="input_value", display_name="Input",
            info="The input value to pass to the chain.", required=True
        ),
        HandleInput(name="llm", display_name="Language Model",
                    input_types=["LanguageModel"], required=True),
    ]

    def invoke_chain(self):
        """
        Invoke the chain to summarize the text.

        Returns:
            Text: The summarized text.
        """
        doc = [
            Document(page_content=self.input_value,
                     metadata={"source": "local"})
        ]
        map_template = """A seguir está um texto/documentos
        {doc}
        Com base neste texto/documento, utilize técnicas de extração e abstração para identificar e condensar as informações mais importantes do texto, foque em palavras-chave, termos técnicos e conceitos principais para garantir que o resumo seja informativo e preciso, evite incluir informações redundantes ou secundárias que não contribuam diretamente para a compreensão do conteúdo central..
        Resposta:"""
        reduce_template = """Você é um assistente especializado em resumos de textos, focado em fornecer resumos sucintos e ricos em conteúdo. Sua prioridade é captar e condensar as informações mais relevantes e importantes, utilizando uma linguagem clara e objetiva.
        Requisitos: 
        1. Resumo Sucinto: O resumo deve ser o mais curto possível em termos de número de caracteres, mantendo a concisão sem sacrificar a clareza.
        2. Conteúdo Rico: O resumo deve conter definições e vocabulário dos tópicos principais abordados no texto original, garantindo que os pontos-chave sejam claramente entendidos.
        A seguir está um conjunto de resumos:
        {doc}
        Pegue-os e transforme-os em um resumo final e consolidado dos temas principais.
        Resposta:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=3000
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="doc",
            return_intermediate_steps=False,
        )
        result = map_reduce_chain.invoke(doc)

        return result['output_text']
