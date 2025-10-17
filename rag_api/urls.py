from django.urls import path
from . import views
from . import test_views
from . import streaming
from . import demo_views

urlpatterns = [
    # Test interface
    path('test/', test_views.test_api_view, name='test-api'),
    path('streaming-demo/', demo_views.streaming_chat_example, name='streaming-chat-demo'),
    
    # Document upload
    path('document-upload/', views.DocumentUploadView.as_view(), name='document-upload'),
    
    # Suggestive questions
    path('suggestive-questions/<uuid:session_id>/', views.SuggestiveQuestionsView.as_view(), name='suggestive-questions'),
    
    # Chat endpoints
    path('chat/', views.ChatView.as_view(), name='chat'),
    path('chat/stream/', streaming.StreamingChatView.as_view(), name='chat-stream'),
    path('session/<uuid:session_id>/messages/', views.SessionMessagesView.as_view(), name='session-messages'),
    
    # File management
    path('file-list/', views.FileListView.as_view(), name='file-list'),
    path('add-additional-files/', views.AddAdditionalFilesView.as_view(), name='add-additional-files'),
    
    # Session management
    path('sessions/', views.SessionListView.as_view(), name='session-list'),
    
    # RAG Agent management
    path('rag-agents/', views.RAGAgentListView.as_view(), name='rag-agents-list'),
    path('rag-agents/<uuid:session_id>/', views.RAGAgentDetailView.as_view(), name='rag-agent-detail'),
    path('rag-agents/<uuid:session_id>/update/', views.RAGAgentUpdateView.as_view(), name='rag-agent-update'),
    path('rag-agents/<uuid:session_id>/delete/', views.RAGAgentDeleteView.as_view(), name='rag-agent-delete'),
]