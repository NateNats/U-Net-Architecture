import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_folder_to_drive(local_folder_path, drive_parent_folder_id=None):
    service = get_drive_service()

    def create_drive_folder(name, parent_id=None):
        metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        if parent_id:
            metadata['parents'] = [parent_id]
        folder = service.files().create(body=metadata, fields='id, webViewLink').execute()
        return folder['id'], folder.get('webViewLink', '')
    
    def upload_file(file_path, parent_id):
        file_name = os.path.basename(file_path)
        file_metadata = {'name': file_name, 'parents': [parent_id]}
        media = MediaFileUpload(file_path, resumable=True)
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"   📄 {file_name}")

    root_name = os.path.basename(os.path.abspath(local_folder_path))
    root_id, root_link = create_drive_folder(root_name, drive_parent_folder_id)
    print(f"\n📁 Folder '{root_name}' dibuat di Drive")
    print(f"🔗 Link: {root_link}\n")

    folder_id_map = {os.path.abspath(local_folder_path): root_id}

    for root, dirs, files in os.walk(local_folder_path):
        abs_root = os.path.abspath(root)
        current_id = folder_id_map[abs_root]

        # Buat subfolder di Drive
        for d in sorted(dirs):
            sub_path = os.path.join(abs_root, d)
            sub_id, _ = create_drive_folder(d, current_id)
            folder_id_map[sub_path] = sub_id
            print(f"   📁 Subfolder '{d}' dibuat")

        # Upload file
        for file in sorted(files):
            upload_file(os.path.join(root, file), current_id)

    print(f"\n✅ Selesai! Semua file terupload ke folder '{root_name}' di Drive.")
    print(f"🔗 Buka di Drive: {root_link}")
    return root_id