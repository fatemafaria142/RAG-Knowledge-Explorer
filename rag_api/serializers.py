import os
from rest_framework import serializers
from .models import RAGSession, Document, ChatMessage, VectorStore, ChunkMetadata


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'filename', 'original_filename', 'document_type', 'file_size', 'upload_date', 'is_processed', 'chunk_count']
        read_only_fields = ['id', 'filename', 'file_size', 'upload_date', 'is_processed', 'chunk_count']


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'message_type', 'content', 'timestamp', 'sources']
        read_only_fields = ['id', 'timestamp']


class RAGSessionSerializer(serializers.ModelSerializer):
    documents = DocumentSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGSession
        fields = ['id', 'name', 'title', 'description', 'instructions', 'created_at', 'updated_at', 'is_active', 'documents', 'message_count']
        read_only_fields = ['id', 'title', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()


class DocumentUploadSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255, help_text="Name for the RAG system")
    description = serializers.CharField(required=False, allow_blank=True, help_text="User-provided description of the RAG system to guide responses")
    instructions = serializers.CharField(required=False, allow_blank=True, help_text="User-provided instructions to guide the RAG agent behavior")
    files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False,
        help_text="Files to upload (.txt, .pdf, or .docx)"
    )
    
    def validate_files(self, value):
        allowed_extensions = ['.txt', '.pdf', '.docx']
        for file in value:
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension not in allowed_extensions:
                raise serializers.ValidationError(f"File {file.name}: Only {', '.join(allowed_extensions)} files are allowed")
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise serializers.ValidationError(f"File {file.name}: Size cannot exceed 10MB")
        return value


class ChatRequestSerializer(serializers.Serializer):
    session_id = serializers.UUIDField(help_text="The UUID of the RAG session")
    message = serializers.CharField(help_text="The user's message or question")


class AddFilesSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    
    def validate_files(self, value):
        allowed_extensions = ['.txt', '.pdf', '.docx']
        for file in value:
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension not in allowed_extensions:
                raise serializers.ValidationError(f"File {file.name}: Only {', '.join(allowed_extensions)} files are allowed")
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise serializers.ValidationError(f"File {file.name}: Size cannot exceed 10MB")
        return value


class ChunkMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChunkMetadata
        fields = ['id', 'document', 'content', 'summary', 'question', 'answer', 'chunk_index', 'metadata']
        read_only_fields = ['id', 'document', 'chunk_index', 'created_at']