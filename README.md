# Tatorte Classifier
## Annotator Tool & Model Creator

### Endpoints
- /api/
    - models/ [GET, POST]
        - GET - Get an ordered list of all the models
        - POST - Create new Model - Input: Metada
    - models/<model_id> [GET, DELETE]
        - GET - Download a saved file from the model
        - DELETE - Delete the model from the database - **Not implemented yet**
    - texts/ [GET, POST]
        - GET - Get an ordered list of all the texts
        - POST - Create a new text - Input: data, categories
    - texts/<text_id> [GET, DELETE, PATCH]
        - GET - Get the document for this text_id
        - DELETE - Delete this document
        - PATCH - Change the categories of the document - Input: categories
    - texts/start=<start>&end=<end> [GET]
        - GET - get an ordered list of the texts from start to end
- / (frontend)
    - / - Index
    - /texts/<page_number> - All texts from (page_number-1)\*100 to page_number\*100
    - /data-checker - Randomly get a text and check its truth
    - /add-data - Add new documents to texts
    - /change-data/<text_id> - Change the categories from the document
    - /new-model - Train new model
    - /models - Get a list of all the models
    - /predictions/<model_id> - Get predictions from the model with model_id on a custom text 