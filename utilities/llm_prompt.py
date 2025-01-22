def get_summary(model: object, results: list, query: str) -> str:
    """Generate a structured summary based on similarity search results.

    This function takes a model, similarity search results, and a query, then
    generates a concise, well-organized summary organized by commodity.

    :param model: The language model object used to generate the summary.
    :param results: A list of similarity search results containing relevant
    data.
    :param query: The user query providing the context for the summary.
    :return: A string containing the refined summary.
    """
    prompt = f"""
        I have performed a similarity search to retrieve relevant data for
        various commodities. Your task is to analyze the similarity search
        results and provide a concise and structured summary for each
        commodity class.

        The summary must:
        - Be organized by commodity.
        - Focus only on the most relevant insights, projections, and trends
          derived from the retrieved results.
        - Exclude any redundant or unnecessary details.

        Here are the similarity search results:
        {results}

        Based on this data, and in response to my query: '{query}', please
        provide a refined summary for each commodity. Ensure the information is
        well-organized, accurate, and directly addresses the query.
    """
    response = model.generate_content(prompt)
    return response.text


def enhance_query(query: str, model: object) -> str:

    prompt = f"""
        Your task is to refine the user's query semantically to make it
        clearer and more meaningful. Avoid adding any irrelevant details
        or altering the core intent of the query.

        Examples:
        - Query: cotton
          Enhanced Query: What are the insights about cotton?

        - Query: wheat consumption region wise
          Enhanced Query: What is the region-wise consumption of wheat?

        - Query: explain oil seeds
          Enhanced Query: What are the trends in oil seeds?

        Based on these examples, refine the following user query:
        {query}
    """
    response = model.generate_content(prompt)
    return response.text
