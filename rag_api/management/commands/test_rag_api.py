import os
from django.core.management.base import BaseCommand
from django.core.files.uploadedfile import SimpleUploadedFile
from rag_api.views import DocumentUploadView
from rest_framework.test import APIRequestFactory

class Command(BaseCommand):
    help = 'Test the RAG API functionality'

    def add_arguments(self, parser):
        parser.add_argument('--name', type=str, help='Name for the RAG session')
        parser.add_argument('--file', type=str, help='Path to the text file to upload')
        parser.add_argument('--action', type=str, choices=['upload', 'list_sessions'], 
                           help='Action to perform')

    def handle(self, *args, **options):
        action = options['action']
        
        if action == 'upload':
            name = options['name']
            file_path = options['file']
            
            if not name or not file_path:
                self.stdout.write(self.style.ERROR('Please provide both name and file path'))
                return
                
            if not os.path.exists(file_path):
                self.stdout.write(self.style.ERROR(f'File not found: {file_path}'))
                return
                
            # Create test request
            factory = APIRequestFactory()
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Create request with file
            file_name = os.path.basename(file_path)
            uploaded_file = SimpleUploadedFile(file_name, file_content, content_type='text/plain')
            
            request = factory.post('/api/document-upload/', 
                                  {'name': name, 'file': uploaded_file}, 
                                  format='multipart')
            
            # Process request
            view = DocumentUploadView.as_view()
            response = view(request)
            
            # Display response
            self.stdout.write(self.style.SUCCESS('Response status: %s' % response.status_code))
            self.stdout.write(self.style.SUCCESS('Response data: %s' % response.data))
            
        elif action == 'list_sessions':
            from rag_api.models import RAGSession
            
            sessions = RAGSession.objects.all()
            self.stdout.write(self.style.SUCCESS(f'Found {sessions.count()} RAG sessions:'))
            
            for session in sessions:
                self.stdout.write('-----------------------------------')
                self.stdout.write(f'ID: {session.id}')
                self.stdout.write(f'Name: {session.name}')
                self.stdout.write(f'Title: {session.title}')
                self.stdout.write(f'Created: {session.created_at}')
                self.stdout.write(f'Documents: {session.documents.count()}')
                self.stdout.write(f'Messages: {session.messages.count()}')