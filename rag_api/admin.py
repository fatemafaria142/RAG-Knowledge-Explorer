from django.contrib import admin
from .models import RAGSession, Document, ChatMessage, VectorStore


@admin.register(RAGSession)
class RAGSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'title', 'created_at', 'updated_at', 'is_active']
    list_filter = ['is_active', 'created_at', 'updated_at']
    search_fields = ['name', 'title']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'session', 'file_size', 'upload_date', 'is_processed']
    list_filter = ['is_processed', 'upload_date']
    search_fields = ['original_filename', 'session__name']
    readonly_fields = ['id', 'upload_date']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'message_type', 'content_preview', 'timestamp']
    list_filter = ['message_type', 'timestamp']
    search_fields = ['content', 'session__name']
    readonly_fields = ['id', 'timestamp']
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content Preview'


@admin.register(VectorStore)
class VectorStoreAdmin(admin.ModelAdmin):
    list_display = ['collection_name', 'session', 'document_count', 'created_at', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['collection_name', 'session__name']
    readonly_fields = ['created_at', 'updated_at']
