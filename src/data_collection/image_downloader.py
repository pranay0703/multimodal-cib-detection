import os
import requests

def download_image(image_url, save_directory='data/images'):
    """
    Downloads an image from a given URL and saves it to a directory.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Extract filename from URL
        filename = os.path.join(save_directory, image_url.split('/')[-1])

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {image_url} to {filename}")
        return filename

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None

if __name__ == '__main__':
    print("This script is for downloading images.")
    
    # Example usage:
    # test_image_url = 'https://www.commoncrawl.org/wp-content/uploads/2021/10/logo-common-crawl.png'
    # download_image(test_image_url)

