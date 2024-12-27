const form = document.getElementById('transaction-form');
const recommendationsDiv = document.getElementById('recommendations');
const recommendationList = document.getElementById('recommendation-list');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Ambil data dari textarea
  const transactionsInput = document.getElementById('transactions').value;
  const transactions = transactionsInput
    .split('\n') // Setiap transaksi dipisahkan dengan baris baru
    .map((line) => line.split(',').map((item) => item.trim())); // Pisahkan dengan koma dan trim

  // Kirim data ke backend
  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transactions }),
    });

    if (response.ok) {
      const data = await response.json();
      showRecommendations(data);
    } else {
      alert('Terjadi kesalahan saat mengambil rekomendasi.');
    }
  } catch (error) {
    alert('Gagal terhubung ke server.');
    console.error(error);
  }
});

function showRecommendations(data) {
  recommendationsDiv.classList.remove('hidden');
  recommendationList.innerHTML = ''; // Kosongkan daftar rekomendasi sebelumnya

  if (data.length === 0) {
    recommendationList.innerHTML = '<li>Tidak ada rekomendasi yang tersedia.</li>';
  } else {
    data.forEach((rule) => {
      const listItem = document.createElement('li');
      listItem.textContent = `Jika pelanggan membeli ${rule.antecedent.join(', ')}, sarankan ${rule.consequent.join(', ')}`;
      recommendationList.appendChild(listItem);
    });
  }
}
