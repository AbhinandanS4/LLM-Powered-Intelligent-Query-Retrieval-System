from django.urls import path
import ingestion_api.views as views

urlpatterns = [
    # Defines the endpoint: POST /api/upload/
    path('upload/', views.upload_document_view, name='upload_document'),
    # Defines the new endpoint: POST /api/query/
    path('query/', views.query_view, name='query_document'),
]