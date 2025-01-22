import os
import time
import hashlib
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


class VectorDB():
    def __init__(self, uploaded_file: UploadedFile):
        self.uploaded_file = uploaded_file
        self.output_file = os.path.join(os.getcwd(), "output.pdf")
        self.pc = Pinecone(api_key=os.getenv("PINE_CONE_API"))

    def get_hash(self, uploaded_file: UploadedFile) -> str:
        """
        convert the pdf file into a hash value
        """
        file_content = uploaded_file.read()
        hash_sha256 = hashlib.sha256(file_content)
        base64_hash = base64.urlsafe_b64encode(hash_sha256.digest())\
            .decode().rstrip('=').replace('_', '-')
        self.index_name = base64_hash.lower()
        return self.index_name

    def save_pdf(self) -> str:
        """
        save pdf
        """
        with open(self.output_file, "wb") as file:
            file.write(self.uploaded_file.getbuffer())

    def chunk(self) -> list:
        """Chunk a PDF document into smaller parts using a text splitter.

        This method saves the current PDF, loads its content, and splits it into # noqa: E501
        smaller chunks based on the specified chunk size and overlap.

        :return: A list of document chunks.
        """
        self.save_pdf()
        loader = PyPDFLoader(self.output_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        docs = text_splitter.split_documents(documents)
        return docs

    def convert_vectors(self) -> object:
        """Convert document chunks into embeddings and load them into a vector index. # noqa: E501

        This function:
        1. Generates document chunks using the `chunk` method.
        2. Converts each chunk into a vector embedding using a multilingual embedding model.
        3. Loads the generated embeddings into a vector database index.

        :return: The loaded vector database index.
        """
        docs = self.chunk()
        data_list = []
        cnt = 1
        for doc in docs:
            data = {
                'id': f"Vec{cnt}",
                'text': doc.page_content
            }
            cnt += 1
            data_list.append(data)
        embeddings_list = []
        prev_index = 0
        if len(data_list) < 96:
            embeddings = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[d['text'] for d in data_list],
                parameters={"input_type": "passage", "truncate": "END"}
            )

            embeddings_list.extend(embeddings)
        else:
            for data in range(96, len(data_list), 96):
                embeddings = self.pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[d['text'] for d in data_list[prev_index:data]],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                embeddings_list.extend(embeddings)
                prev_index = data

                if prev_index + 96 > len(data_list):
                    embeddings = self.pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[d['text'] for d in data_list[prev_index:len(data_list)]],  # noqa: E501
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    embeddings_list.extend(embeddings)
        self.create_index()
        index = self.load_vectors(data_list, embeddings_list)
        return index

    def create_index(self) -> None:
        """Create a serverless vector index if it does not exist.

        This function checks if the specified index exists. If it does not, it creates a # noqa: E501
        new serverless vector index with the given configuration. The function also waits
        until the index is ready to ensure it's fully operational before proceeding.

        :return: None
        """
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(5)

    def load_vectors(self, data_list: list, embeddings: list) -> object:
        """Load vector embeddings into the specified index.

        This function takes a list of data objects and their corresponding embeddings, # noqa: E501
        formats them into records, and upserts them into a vector index. It then waits
        for the upsert process to complete and prints the index statistics.

        :param data_list: A list of dictionaries containing document IDs and text.
        :param embeddings: A list of embeddings corresponding to the documents.
        :return: The vector index object.
        """

        index = self.pc.Index(self.index_name)
        records = []

        for d, e in zip(data_list, embeddings):
            records.append({
                "id": d['id'],
                "values": e.values,
                "metadata": {'text': d['text']}
            })

        index.upsert(
            vectors=records,
            namespace=self.index_name
        )

        time.sleep(20)
        return index

    def get_results(self, index: object, query: str) -> list:
        """Retrieve a summary by querying the vector index for similar documents. # noqa: E501

        This function converts a query into a numerical vector, searches the vector
        index for the most similar embeddings, and extracts the associated metadata
        (e.g., text) from the results.

        :param index: The vector index object to search.
        :param query: The user query to be converted into a vector and searched.
        :return: A list of text snippets from the most similar documents.
        """
        query_embedding = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )

        results = index.query(
            namespace=self.index_name,
            vector=query_embedding[0].values,
            top_k=10,
            include_values=False,
            include_metadata=True
        )

        output = [result['metadata']['text'] for result in results['matches']]
        return output

    def delete_index(self, index: str) -> None:
        """Delete a specified Pinecone index.

        This function deletes the given Pinecone index if it exists.

        :param index: The name of the Pinecone index to be deleted.
        :return: None
        """
        if index:
            self.pc.delete_index(index)

    def validate_index(self, hash_key: str) -> bool:
        """Check if a specified Pinecone index exists.

        This function checks whether the given index (identified by the
        hash key) exists in the Pinecone service.

        :param hash_key: The unique identifier (hash key) of the Pinecone
        index to check.
        :return: True if the index exists, False otherwise.
        """
        if self.pc.has_index(hash_key):
            return True
