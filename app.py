"""Compatibility entry point: python app.py."""
import uvicorn
if __name__ == '__main__':
    uvicorn.run('image_generator.app:app', host='127.0.0.1', port=8003)
