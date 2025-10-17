import os
import uuid
from django.shortcuts import get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .models import RAGSession, Document, ChatMessage, VectorStore, ChunkMetadata
from .serializers import (
    RAGSessionSerializer, DocumentSerializer, ChatMessageSerializer,
    DocumentUploadSerializer, ChatRequestSerializer, 
    AddFilesSerializer
)
from .rag_system import get_rag_system


class DocumentUploadView(APIView):
    """Upload documents and create RAG session"""
    
    @swagger_auto_schema(
        operation_description="Upload multiple documents and create a new RAG session with user-provided instructions and description",
        manual_parameters=[
            openapi.Parameter(
                name='name',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description='Name for the RAG session'
            ),
            openapi.Parameter(
                name='description',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=False,
                description='User-provided description to guide RAG responses and provide context'
            ),
            openapi.Parameter(
                name='instructions',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=False,
                description='User-provided instructions to guide the RAG agent behavior and response style'
            ),
            openapi.Parameter(
                name='files',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(type=openapi.TYPE_FILE),
                required=True,
                description='Files to upload (.txt, .pdf, or .docx)',
                collection_format='multi'
            ),
        ],
        responses={
            status.HTTP_201_CREATED: openapi.Response(
                description="Document uploaded and RAG session created successfully",
                schema=RAGSessionSerializer
            ),
            status.HTTP_400_BAD_REQUEST: "Invalid input"
        },
        tags=['Documents'],
        consumes=['multipart/form-data']
    )
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Extract data
            name = serializer.validated_data['name']
            files = serializer.validated_data['files']
            description = serializer.validated_data.get('description', '')
            instructions = serializer.validated_data.get('instructions', '')
            
            # Get RAG system
            rag_system = get_rag_system()
            
            # Process files and collect text content
            documents_content = []
            document_types = []
            all_text_content = ""
            
            for file in files:
                # Determine file type based on extension
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension == '.txt':
                    document_type = 'txt'
                    # Read text file content
                    file_content = file.read().decode('utf-8')
                    text_content = file_content
                elif file_extension == '.pdf':
                    document_type = 'pdf'
                    # Read binary content for PDF
                    file_content = file.read()
                    # Extract text from PDF
                    text_content = rag_system.extract_text_from_file(file_content, document_type)
                elif file_extension == '.docx':
                    document_type = 'docx'
                    # Read binary content for DOCX
                    file_content = file.read()
                    # Extract text from DOCX
                    text_content = rag_system.extract_text_from_file(file_content, document_type)
                else:
                    return Response({"error": f"Unsupported file type for {file.name}"}, status=status.HTTP_400_BAD_REQUEST)
                
                if not text_content:
                    return Response({
                        'error': f'Failed to extract text from {file.name}'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                documents_content.append(text_content)
                document_types.append(document_type)
                all_text_content += text_content + "\n\n"
            
            # Generate title from combined content
            title = rag_system.generate_title_from_content(all_text_content[:2000])  # Use first 2000 chars for title generation
            
            # Create RAG session with description and instructions
            rag_session = RAGSession.objects.create(
                name=name,
                title=title,
                description=description,
                instructions=instructions
            )
            
            # Create document records
            created_documents = []
            for i, file in enumerate(files):
                document = Document.objects.create(
                    session=rag_session,
                    filename=f"{rag_session.id}_{file.name}",
                    original_filename=file.name,
                    file_size=file.size,
                    document_type=document_types[i],
                    content=documents_content[i] if document_types[i] == 'txt' else ''  # Store text content only for txt files
                )
                created_documents.append(document)
            
            # Process documents in vector store with enhanced metadata
            vector_success, chunk_metadata = rag_system.process_documents(
                documents_content, 
                str(rag_session.id),
                created_documents,
                document_types
            )
            
            if vector_success:
                # Create vector store record
                VectorStore.objects.create(
                    session=rag_session,
                    collection_name=f"rag_collection_{str(rag_session.id).replace('-', '_')}",
                    document_count=len(files)
                )
                
                # Create chunk metadata records and update documents
                document_chunk_counts = {}
                for chunk in chunk_metadata:
                    doc_index = chunk['document_index']
                    if doc_index < len(created_documents):
                        document = created_documents[doc_index]
                        
                        # Create chunk metadata record
                        ChunkMetadata.objects.create(
                            document=document,
                            content=chunk['content'],
                            summary=chunk['summary'],
                            question=chunk['question'],
                            answer=chunk['answer'],
                            chunk_index=chunk['chunk_index'],
                            metadata={"document_index": doc_index}
                        )
                        
                        # Count chunks per document
                        document_chunk_counts[str(document.id)] = document_chunk_counts.get(str(document.id), 0) + 1
                
                # Mark documents as processed and update chunk counts
                for document in created_documents:
                    document.is_processed = True
                    document.chunk_count = document_chunk_counts.get(str(document.id), 0)
                    document.save()
            else:
                # Still save the documents but mark as not processed
                for document in created_documents:
                    document.is_processed = False
                    document.save()
            
            # Return session info
            session_serializer = RAGSessionSerializer(rag_session)
            return Response({
                'session': session_serializer.data,
                'message': f'{len(files)} documents uploaded and processed successfully'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({
                'error': f'Failed to process document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SuggestiveQuestionsView(APIView):
    """Generate suggestive questions for a RAG session"""
    
    @swagger_auto_schema(
        operation_description="Generate suggested questions based on document content for a specific session",
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Questions generated successfully",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid'),
                        'questions': openapi.Schema(
                            type=openapi.TYPE_ARRAY, 
                            items=openapi.Schema(type=openapi.TYPE_STRING),
                            description='List of suggested questions'
                        ),
                    }
                )
            ),
            status.HTTP_404_NOT_FOUND: "Session not found",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "Failed to generate questions"
        },
        tags=['Chat']
    )
    def get(self, request, session_id):
        try:
            print(f"Generating suggestive questions for session: {session_id}")
            # Verify session exists
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Check if session has documents
            document_count = session.documents.count()
            print(f"Session has {document_count} documents")
            
            if document_count == 0:
                return Response({
                    'error': 'No documents found in this session. Please upload documents first.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get processed document count
            processed_count = session.documents.filter(is_processed=True).count()
            print(f"Session has {processed_count} processed documents")
            
            if processed_count == 0:
                return Response({
                    'error': 'Documents are still being processed. Please try again later.'
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Generate questions
            rag_system = get_rag_system()
            if rag_system is None:
                return Response({
                    'error': 'RAG system not initialized properly'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            questions = rag_system.generate_suggestive_questions(str(session_id))
            
            print(f"Generated questions: {questions}")
            return Response({
                'session_id': session_id,
                'questions': questions
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({
                'error': f'Failed to generate questions: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatView(APIView):
    """Handle chat messages"""
    
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['session_id', 'message'],
            properties={
                'session_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='The ID of the RAG session'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='The message to process')
            }
        ),
        operation_description="Send a message to an existing RAG session",
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Message processed successfully",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'user_message': openapi.Schema(type=openapi.TYPE_OBJECT),
                        'assistant_message': openapi.Schema(type=openapi.TYPE_OBJECT, 
                            properties={
                                'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid'),
                                'message_type': openapi.Schema(type=openapi.TYPE_STRING),
                                'content': openapi.Schema(type=openapi.TYPE_STRING),
                                'timestamp': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                                'sources': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING))
                            }
                        )
                    }
                )
            ),
            status.HTTP_400_BAD_REQUEST: "Invalid input or session not found"
        },
        tags=['Chat']
    )
    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            session_id = serializer.validated_data['session_id']
            message_content = serializer.validated_data['message']
            
            # Get session
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Get chat history
            chat_history = ChatMessage.objects.filter(session=session).order_by('timestamp')
            history_data = ChatMessageSerializer(chat_history, many=True).data
            
            # Save user message
            user_message = ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message_content
            )
            
            # Use session values for instructions and description
            instructions = session.instructions
            description = session.description
            
            # Combine instructions and description if available
            combined_instructions = ""
            if instructions and description:
                combined_instructions = f"Description: {description}\n\nInstructions: {instructions}"
            elif instructions:
                combined_instructions = instructions
            elif description:
                combined_instructions = f"Description: {description}"
            
            # Process through RAG system with React agent
            try:
                rag_system = get_rag_system()
                if rag_system is None:
                    raise Exception("RAG system initialization failed")
                    
                # Explicitly set stream=False to ensure we get a dictionary response, not a generator
                rag_response = rag_system.chat(
                    session_id=str(session_id), 
                    question=message_content, 
                    chat_history=history_data,
                    instructions=combined_instructions,
                    stream=False
                )
                
                if not isinstance(rag_response, dict):
                    raise Exception(f"Expected dictionary response, got {type(rag_response).__name__}")
                    
            except Exception as e:
                print(f"❌ RAG system error: {e}")
                raise Exception(f"Failed to process chat: {e}")
            
            # Save assistant message
            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type='assistant',
                content=rag_response['answer'],
                sources=rag_response['sources']
            )
            
            # Update session timestamp
            session.save()  # This will update the updated_at field
            
            return Response({
                'user_message': ChatMessageSerializer(user_message).data,
                'assistant_message': ChatMessageSerializer(assistant_message).data
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to process chat: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionMessagesView(APIView):
    """Get messages for a session"""
    
    @swagger_auto_schema(
        operation_description="Get all messages for a specific session",
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Messages retrieved successfully",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid'),
                        'messages': openapi.Schema(
                            type=openapi.TYPE_ARRAY, 
                            items=openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid'),
                                    'message_type': openapi.Schema(type=openapi.TYPE_STRING),
                                    'content': openapi.Schema(type=openapi.TYPE_STRING),
                                    'timestamp': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                                    'sources': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING))
                                }
                            )
                        ),
                    }
                )
            ),
            status.HTTP_404_NOT_FOUND: "Session not found"
        },
        tags=['Chat']
    )
    def get(self, request, session_id):
        try:
            session = get_object_or_404(RAGSession, id=session_id)
            messages = ChatMessage.objects.filter(session=session).order_by('timestamp')
            serializer = ChatMessageSerializer(messages, many=True)
            
            return Response({
                'session_id': session_id,
                'messages': serializer.data
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to get messages: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FileListView(APIView):
    """List all files in vector database"""
    
    def get(self, request):
        try:
            # Get all active sessions with their documents
            sessions = RAGSession.objects.filter(is_active=True).prefetch_related('documents')
            
            file_list = []
            for session in sessions:
                for document in session.documents.all():
                    file_list.append({
                        'session_id': str(session.id),
                        'session_name': session.name,
                        'document_id': str(document.id),
                        'filename': document.original_filename,
                        'file_size': document.file_size,
                        'upload_date': document.upload_date,
                        'is_processed': document.is_processed
                    })
            
            return Response({
                'files': file_list,
                'total_count': len(file_list)
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to list files: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# FileDeleteView has been removed as requested


class AddAdditionalFilesView(APIView):
    """Add additional files to existing RAG session"""
    
    @swagger_auto_schema(
        operation_description="Add additional files to an existing RAG session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                format='uuid',
                required=True,
                description='The ID of the existing RAG session'
            ),
            openapi.Parameter(
                name='files',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description='Files to upload (.txt, .pdf, or .docx)',
                collection_format='multi'
            ),
        ],
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Files added to session successfully"
            ),
            status.HTTP_400_BAD_REQUEST: "Invalid input or session not found",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "Failed to process files"
        },
        tags=['Documents'],
        consumes=['multipart/form-data']
    )
    def post(self, request):
        serializer = AddFilesSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            session_id = serializer.validated_data['session_id']
            files = serializer.validated_data['files']
            
            # Get session
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Process each file
            documents_content = []
            created_documents = []
            document_types = []
            rag_system = get_rag_system()
            
            for file in files:
                # Determine file type based on extension
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension == '.txt':
                    document_type = 'txt'
                    # Read text file content
                    file_content = file.read().decode('utf-8')
                    text_content = file_content
                elif file_extension == '.pdf':
                    document_type = 'pdf'
                    # Read binary content for PDF
                    file_content = file.read()
                    # Extract text from PDF
                    text_content = rag_system.extract_text_from_file(file_content, document_type)
                elif file_extension == '.docx':
                    document_type = 'docx'
                    # Read binary content for DOCX
                    file_content = file.read()
                    # Extract text from DOCX
                    text_content = rag_system.extract_text_from_file(file_content, document_type)
                else:
                    continue  # Skip unsupported files
                
                if not text_content:
                    continue  # Skip if text extraction failed
                
                documents_content.append(text_content)
                document_types.append(document_type)
                
                # Create document record
                document = Document.objects.create(
                    session=session,
                    filename=f"{session.id}_{file.name}",
                    original_filename=file.name,
                    file_size=file.size,
                    document_type=document_type,
                    content=text_content if document_type == 'txt' else ''  # Store text content only for txt files
                )
                created_documents.append(document)
            
            # Add documents to existing vector store with enhanced metadata
            success, chunk_metadata = rag_system.process_documents(
                documents_content, 
                str(session_id),
                created_documents,
                document_types
            )
            
            if success:
                # Create chunk metadata and mark documents as processed
                document_chunk_counts = {}
                for chunk in chunk_metadata:
                    doc_index = chunk['document_index']
                    if doc_index < len(created_documents):
                        document = created_documents[doc_index]
                        
                        # Create chunk metadata record
                        ChunkMetadata.objects.create(
                            document=document,
                            content=chunk['content'],
                            summary=chunk['summary'],
                            question=chunk['question'],
                            answer=chunk['answer'],
                            chunk_index=chunk['chunk_index'],
                            metadata={"document_index": doc_index}
                        )
                        
                        # Count chunks per document
                        document_chunk_counts[str(document.id)] = document_chunk_counts.get(str(document.id), 0) + 1
                
                # Mark documents as processed and update chunk counts
                for document in created_documents:
                    document.is_processed = True
                    document.chunk_count = document_chunk_counts.get(str(document.id), 0)
                    document.save()
                
                # Update vector store document count
                vector_store = VectorStore.objects.get(session=session)
                vector_store.document_count = Document.objects.filter(session=session).count()
                vector_store.save()
                
                # Update session timestamp
                session.save()
                
                return Response({
                    'message': f'Successfully added {len(files)} files to the RAG session',
                    'added_files': [doc.original_filename for doc in created_documents]
                })
            else:
                # Clean up if vector store update failed
                for document in created_documents:
                    document.delete()
                
                return Response({
                    'error': 'Failed to add files to vector store'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except Exception as e:
            return Response({
                'error': f'Failed to add files: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionListView(generics.ListAPIView):
    """List all RAG sessions"""
    queryset = RAGSession.objects.filter(is_active=True)
    serializer_class = RAGSessionSerializer
    
    @swagger_auto_schema(
        operation_description="List all active RAG sessions",
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="List of RAG sessions retrieved successfully",
                schema=openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid'),
                            'name': openapi.Schema(type=openapi.TYPE_STRING),
                            'title': openapi.Schema(type=openapi.TYPE_STRING),
                            'documents': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_OBJECT)),
                            'message_count': openapi.Schema(type=openapi.TYPE_INTEGER)
                        }
                    )
                )
            )
        },
        tags=['Sessions']
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


class RAGAgentListView(APIView):
    """List all RAG agents with their details"""
    
    def get(self, request):
        try:
            sessions = RAGSession.objects.filter(is_active=True).prefetch_related('documents', 'messages')
            
            agents = []
            for session in sessions:
                # Count documents and messages
                document_count = session.documents.count()
                message_count = session.messages.count()
                
                # Get last activity
                last_message = session.messages.order_by('-timestamp').first()
                last_activity = last_message.timestamp if last_message else session.created_at
                
                agents.append({
                    'id': str(session.id),
                    'name': session.name,
                    'title': session.title,
                    'description': session.description or 'No description available',
                    'created_at': session.created_at,
                    'updated_at': session.updated_at,
                    'last_activity': last_activity,
                    'document_count': document_count,
                    'message_count': message_count,
                    'is_active': session.is_active
                })
            
            return Response({
                'agents': agents,
                'total_count': len(agents)
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to get RAG agents: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGAgentDetailView(APIView):
    """Get detailed information about a specific RAG agent"""
    
    def get(self, request, session_id):
        try:
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Get all documents
            documents = Document.objects.filter(session=session)
            doc_data = []
            for doc in documents:
                doc_data.append({
                    'id': str(doc.id),
                    'filename': doc.original_filename,
                    'file_size': doc.file_size,
                    'upload_date': doc.upload_date,
                    'is_processed': doc.is_processed,
                    'content_preview': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content
                })
            
            # Get recent messages
            recent_messages = ChatMessage.objects.filter(session=session).order_by('-timestamp')[:10]
            message_data = ChatMessageSerializer(recent_messages, many=True).data
            
            # Get vector store info
            try:
                vector_store = VectorStore.objects.get(session=session)
                vector_info = {
                    'collection_name': vector_store.collection_name,
                    'document_count': vector_store.document_count,
                    'created_at': vector_store.created_at
                }
            except VectorStore.DoesNotExist:
                vector_info = None
            
            return Response({
                'session': RAGSessionSerializer(session).data,
                'documents': doc_data,
                'recent_messages': message_data,
                'vector_store': vector_info,
                'statistics': {
                    'total_documents': len(doc_data),
                    'total_messages': ChatMessage.objects.filter(session=session).count(),
                    'processed_documents': sum(1 for doc in doc_data if doc['is_processed']),
                }
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to get RAG agent details: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGAgentDeleteView(APIView):
    """Delete a RAG agent and all its data"""
    
    def delete(self, request, session_id):
        try:
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Get all data that will be deleted for confirmation
            documents = Document.objects.filter(session=session)
            messages = ChatMessage.objects.filter(session=session)
            
            # Delete from vector database
            get_rag_system().delete_session_vector_store(str(session_id))
            
            # Delete vector store record
            VectorStore.objects.filter(session=session).delete()
            
            # Delete documents and messages (CASCADE should handle this, but explicit is better)
            documents.delete()
            messages.delete()
            
            # Delete session
            session_name = session.name
            session.delete()
            
            return Response({
                'message': f'RAG agent "{session_name}" and all associated data deleted successfully',
                'deleted_items': {
                    'documents': documents.count(),
                    'messages': messages.count(),
                    'vector_data': 'Cleared from vector database'
                }
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to delete RAG agent: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGAgentUpdateView(APIView):
    """Update RAG agent information"""
    
    @swagger_auto_schema(
        operation_description="Update a RAG agent's instructions and description",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'description': openapi.Schema(type=openapi.TYPE_STRING, description='New description to guide RAG responses'),
                'instructions': openapi.Schema(type=openapi.TYPE_STRING, description='New instructions for the RAG agent behavior')
            }
        ),
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="RAG agent updated successfully",
                schema=RAGSessionSerializer
            ),
            status.HTTP_404_NOT_FOUND: "Session not found",
            status.HTTP_400_BAD_REQUEST: "Invalid input"
        },
        tags=['Sessions']
    )
    def put(self, request, session_id):
        try:
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Update basic session info - focusing on description and instructions
            if 'description' in request.data:
                session.description = request.data['description']
            if 'instructions' in request.data:
                session.instructions = request.data['instructions']
            # Still allow updating name and title if provided
            if 'name' in request.data:
                session.name = request.data['name']
            if 'title' in request.data:
                session.title = request.data['title']
            # Session ID is already captured from the URL parameter, no need to get it from request.data
            
            session.save()
            
            return Response({
                'message': 'RAG agent updated successfully',
                'session': RAGSessionSerializer(session).data
            })
            
        except Exception as e:
            return Response({
                'error': f'Failed to update RAG agent: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
