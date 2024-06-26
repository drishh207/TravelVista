document.addEventListener('DOMContentLoaded', () => {
    const searchButton = document.getElementById('search-btn');
    const searchBar = document.getElementById('search-bar');

    searchButton.addEventListener('click', () => {
        const query = searchBar.value;
        if (query) {
            fetch('http://127.0.0.1:9001/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: query }),
            })
            .then(response => response.json())
            .then(data => {
                if (Array.isArray(data.results)) {
                    const params = new URLSearchParams();
                    params.append('results', JSON.stringify(data.results));
                    window.location.href = `city.html?${params.toString()}`;
                } else {
                    console.error('Unexpected data format:', data);
                }
            })
            .catch(error => console.error('Error:', error));
        }
    });
});

