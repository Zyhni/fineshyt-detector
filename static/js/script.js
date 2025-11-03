let currentMusic = null;

// Drag and drop functionality
const uploadArea = document.getElementById('uploadArea');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
});

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

function handleImageFile(file) {
    if (!file.type.match('image.*')) {
        alert('UPLOAD FOTO BENERAN!');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('uploadArea').style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.error) {
            alert(data.error);
            resetAnalysis();
            return;
        }

        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        alert('ERROR COK! COBA LAGI!');
        resetAnalysis();
    });
}

function displayResults(data) {
    document.getElementById('previewImage').src = data.image;

    const scoreElement = document.getElementById('scoreValue');
    const scoreCircle = document.getElementById('scoreCircle');
    animateValue(scoreElement, 0, data.score, 1000);
    scoreCircle.className = 'score-circle ' + data.category;
    
    // Update image overlay
    document.getElementById('imageOverlay').textContent = data.category.toUpperCase() + ' FINESHYT';
    
    // Update feedback
    document.getElementById('feedbackText').textContent = data.feedback;
    
    // Update factors list
    const factorsList = document.getElementById('factorsList');
    factorsList.innerHTML = '';
    data.factors.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
    });
    
    // Update music player dengan Spotify Embed (randomize client-side jika server kasih same)
    const musicInfo = document.getElementById('musicInfo');
    const spotifyPlayer = document.getElementById('spotifyPlayer');
    
    if (data.music) {
        musicInfo.innerHTML = `
            <strong>${data.music.title}</strong><br>
            🎶 Mood: ${data.music.mood}
        `;
        if (currentMusic && currentMusic.spotify_id === data.music.spotify_id) {
        }
        
        spotifyPlayer.innerHTML = `
            <iframe src="https://open.spotify.com/embed/track/${data.music.spotify_id}" 
                    width="100%" height="80" frameborder="0" 
                    allowtransparency="true" allow="encrypted-media"
                    style="border-radius: 12px; margin-top: 10px;">
            </iframe>
        `;
        
        currentMusic = data.music;
    } else {
        spotifyPlayer.innerHTML = '';
        musicInfo.innerHTML = '';
        musicReason.textContent = '';
        currentMusic = null;
    }
    
    // Show result section
    document.getElementById('resultSection').style.display = 'block';
}

function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function resetAnalysis() {
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInput').value = '';
    
    // Clear Spotify player
    document.getElementById('spotifyPlayer').innerHTML = '';
    
    currentMusic = null;
}

// Add keyboard shortcut
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        resetAnalysis();
    }
});

console.log('🎵 FINESHYT DETECTOR UDH SIAP!');