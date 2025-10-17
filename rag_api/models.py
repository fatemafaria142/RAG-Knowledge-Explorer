from django.db import models
from django.contrib.auth.models import User
import uuid


class RAGSession(models.Model):
    """Model to manage RAG sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="User-provided name for the RAG")
    title = models.CharField(max_length=255, blank=True, help_text="Auto-generated title from document")
    description = models.TextField(blank=True, null=True, help_text="Description of the RAG session")
    instructions = models.TextField(blank=True, null=True, help_text="Instructions to guide the RAG agent behavior")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.name} - {self.title}"


class Document(models.Model):
    """Model to store document information"""
    DOCUMENT_TYPES = (
        ('txt', 'Text'),
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(RAGSession, on_delete=models.CASCADE, related_name='documents')
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_size = models.IntegerField()
    content = models.TextField()
    document_type = models.CharField(max_length=10, choices=DOCUMENT_TYPES, default='txt')
    upload_date = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    vector_store_id = models.CharField(max_length=255, blank=True, null=True)
    chunk_count = models.IntegerField(default=0, help_text="Number of chunks created from this document")
    
    class Meta:
        ordering = ['-upload_date']
    
    def __str__(self):
        return f"{self.original_filename} - {self.session.name}"


class ChatMessage(models.Model):
    """Model to store chat messages"""
    MESSAGE_TYPES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(RAGSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    sources = models.JSONField(default=list, blank=True)  # Store source document references
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."


class VectorStore(models.Model):
    """Model to track vector store collections"""
    session = models.OneToOneField(RAGSession, on_delete=models.CASCADE, related_name='vector_store')
    collection_name = models.CharField(max_length=255, unique=True)
    document_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Vector Store: {self.collection_name}"


class ChunkMetadata(models.Model):
    """Model to store metadata for document chunks including Q&A pairs and summaries"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField(help_text="The actual chunk content")
    summary = models.TextField(blank=True, null=True, help_text="Summary of the chunk content")
    question = models.TextField(blank=True, null=True, help_text="Generated question for this chunk")
    answer = models.TextField(blank=True, null=True, help_text="Generated answer for the question")
    chunk_index = models.IntegerField(help_text="Index of this chunk in the document")
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional metadata for the chunk")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['document', 'chunk_index']
        unique_together = ('document', 'chunk_index')
    
    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.original_filename}"
