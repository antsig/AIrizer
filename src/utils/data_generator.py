import json
import os
import random

def generate_synthetic_data(num_docs=50, output_dir="data/synthetic"):
    os.makedirs(output_dir, exist_ok=True)
    
    docs = []
    
    for i in range(1, num_docs + 1):
        category = random.choice(["Laporan", "Surat Resmi", "Berita", "Invoice"])
        
        if category == "Laporan":
            subj = random.choice(["Proyek IT", "Keuangan Bulanan", "Audit Internal", "Pemasaran", "Evaluasi Kinerja"])
            suff = random.choice(["Q1 2024", "Bulan Maret", "Tahunan", "Fase 1", "Semester Ganjil"])
            title = f"Laporan {subj} {suff}"
            
            v1 = random.choice(["pengembangan aplikasi", "penjualan produk", "audit sistem", "kampanye iklan"])
            v2 = random.choice(["80%", "target awal", "tahap final", "Rp 500 Juta"])
            v3 = random.choice(["server", "distribusi", "keamanan", "konversi"])
            v4 = random.choice(["Andi", "Budi", "Siti", "Tim Support"])
            v5 = random.choice(["Jumat depan", "akhir bulan", "tanggal 15", "Senin pagi"])
            
            content = f"Laporan ini membahas mengenai progres {v1}. Hingga saat ini, pencapaian telah mencapai {v2}. Namun, terdapat beberapa kendala di bagian {v3}. {v4} ditugaskan untuk menyelesaikan masalah ini sebelum {v5}. Secara keseluruhan, kondisi masih terkendali."
            expected_action_items = [f"{v4} -> menyelesaikan masalah ini -> {v5}"]
            
        elif category == "Surat Resmi":
            subj = random.choice(["Jadwal Cuti Bersama", "Perubahan Jam Kerja", "Protokol Kesehatan Baru", "Pemeliharaan Server"])
            title = f"Surat Edaran: {subj}"
            
            v1 = random.choice(["jadwal baru", "aturan terbaru", "kebijakan perusahaan"])
            v2 = random.choice(["Senin depan", "awal bulan", "hari ini"])
            v4 = random.choice(["HRD", "Manager", "Admin", "Sekretaris"])
            v5 = random.choice(["besok", "nanti sore", "tanggal 10"])
            
            content = f"Diberitahukan kepada seluruh karyawan mengenai {v1}. Keputusan ini efektif mulai {v2}. Semua pihak diharap mematuhi aturan ini. {v4} harus menyebarkan informasi ini ke seluruh divisi maksimal {v5}. Terima kasih atas kerjasamanya."
            expected_action_items = [f"{v4} -> menyebarkan informasi ini ke seluruh divisi -> {v5}"]
            
        elif category == "Invoice":
            subj = random.choice(["Server", "Lisensi Software", "Perlengkapan Kantor", "Jasa Konsultan"])
            suff = random.choice(["#INV-001", "#INV-002", "#INV-003", "#INV-004"])
            title = f"Invoice Pembelian {subj} {suff}"
            
            v1 = random.choice(["pembelian alat", "perpanjangan lisensi", "layanan premium"])
            v2 = random.choice(["Rp 10.000.000", "Rp 5.500.000", "Rp 25.000.000"])
            v4 = random.choice(["Bagian Keuangan", "Bendahara", "Divisi Purchasing", "Akuntan"])
            v5 = random.choice(["tanggal 25", "akhir bulan ini", "Jumat"])
            
            content = f"Tagihan untuk {v1}. Total biaya yang harus dibayarkan adalah {v2}. Pembayaran dapat dilakukan melalui transfer ke rekening Bank BCA 123456. {v4} harus menyelesaikan pembayaran ini paling lambat {v5}."
            expected_action_items = [f"{v4} -> menyelesaikan pembayaran ini -> {v5}"]
            
        else: # Berita
            subj = random.choice(["Perkembangan AI Terbaru", "Startup Lokal Mendapat Pendanaan", "Tren Cloud Computing 2024", "Regulasi Data Pribadi Baru"])
            title = f"Berita: {subj}"
            
            v1 = random.choice(["teknologi baru telah dirilis", "regulasi ketat diterapkan", "investasi meningkat tajam"])
            v2 = random.choice(["mengubah pasar", "menciptakan peluang baru", "meningkatkan efisiensi"])
            v3 = random.choice(["bersiap", "memperbarui sistem", "melakukan pelatihan"])
            
            content = f"Hari ini dilaporkan bahwa {v1}. Hal ini diprediksi akan {v2}. Para ahli menyarankan agar industri {v3}. Tidak ada tindakan mendesak, namun perusahaan harus mulai beradaptasi dengan perubahan ini."
            expected_action_items = []

        doc = {
            "id": f"doc_{i:03d}",
            "title": title,
            "category": category,
            "content": content,
            "expected_action_items": expected_action_items,
            "expected_keywords": []
        }
        docs.append(doc)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)
        
    print(f"Berhasil membuat dataset sintetik sebanyak {len(docs)} dokumen di {output_dir}")

if __name__ == "__main__":
    generate_synthetic_data(num_docs=50)
