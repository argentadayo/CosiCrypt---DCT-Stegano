from flask import Flask, render_template, request, send_file, redirect, flash
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import io

app = Flask(
    __name__,
    template_folder='.',       # Cari index.html di folder sekarang
    static_folder='.',         # Cari styles.css di folder sekarang
    static_url_path=''         # Akses CSS di URL '/styles.css'
)
app.secret_key = 'replace-with-your-secret-key'

# JPEG-like luminance quantization table for 8x8 blocks
Q_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=float)

TERMINATOR = "***END***"
# Reduce embedding capacity: use only 4 DCT positions per block instead of 8
EMBED_POSITIONS = [(1, 2), (2, 1), (1, 3), (3, 1)]  # now 4 positions
REP_FACTOR = 3


def dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(coeffs: np.ndarray) -> np.ndarray:
    return idct(idct(coeffs, axis=0, norm='ortho'), axis=1, norm='ortho')


def embed_message(img: Image.Image, message: str) -> Image.Image:
    gray = img.convert('L')
    arr = np.array(gray, dtype=float)
    h, w = arr.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')

    full = message + TERMINATOR
    bits = ''.join(f"{ord(c):08b}" for c in full)
    bits_rep = ''.join(bit * REP_FACTOR for bit in bits)

    blocks = (arr_padded.shape[0] // 8) * (arr_padded.shape[1] // 8)
    capacity_bits = blocks * len(EMBED_POSITIONS)
    max_chars = (capacity_bits // REP_FACTOR) // 8
    if len(bits_rep) > capacity_bits:
        raise ValueError(f"Pesan terlalu panjang: maksimal {max_chars} karakter.")

    bit_idx = 0
    for i in range(0, arr_padded.shape[0], 8):
        for j in range(0, arr_padded.shape[1], 8):
            block = arr_padded[i:i+8, j:j+8]
            coeffs = dct2(block)
            qcoeffs = np.round(coeffs / Q_TABLE).astype(int)
            for pos in EMBED_POSITIONS:
                if bit_idx < len(bits_rep):
                    qcoeffs[pos] = (qcoeffs[pos] & ~1) | int(bits_rep[bit_idx])
                    bit_idx += 1
                else:
                    break
            coeffs_mod = qcoeffs * Q_TABLE
            arr_padded[i:i+8, j:j+8] = idct2(coeffs_mod)
            if bit_idx >= len(bits_rep):
                break
        if bit_idx >= len(bits_rep):
            break

    stego_arr = np.clip(arr_padded[:h, :w], 0, 255).astype(np.uint8)
    return Image.fromarray(stego_arr).convert('RGB')


def extract_message(img: Image.Image) -> str:
    gray = img.convert('L')
    arr = np.array(gray, dtype=float)
    h, w = arr.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')

    bits = []
    for i in range(0, arr_padded.shape[0], 8):
        for j in range(0, arr_padded.shape[1], 8):
            block = arr_padded[i:i+8, j:j+8]
            coeffs = dct2(block)
            qcoeffs = np.round(coeffs / Q_TABLE).astype(int)
            for pos in EMBED_POSITIONS:
                bits.append(str(qcoeffs[pos] & 1))

    orig_bits = []
    for k in range(0, len(bits), REP_FACTOR):
        group = bits[k:k+REP_FACTOR]
        if len(group) < REP_FACTOR:
            break
        orig_bits.append('1' if group.count('1') > REP_FACTOR//2 else '0')

    chars = []
    for k in range(0, len(orig_bits), 8):
        byte = orig_bits[k:k+8]
        if len(byte) < 8:
            break
        val = int(''.join(byte), 2)
        chars.append(chr(val))
        if ''.join(chars).endswith(TERMINATOR):
            break

    return ''.join(chars).replace(TERMINATOR, '')


@app.route('/', methods=['GET', 'POST'])
def index():
    hidden = None

    # Encode
    if request.method == 'POST' and 'encode' in request.form:
        file = request.files.get('image')
        msg = request.form.get('message', '')
        if not file or not msg:
            flash('Gambar dan pesan wajib diisi!')
            return redirect(request.url)
        img = Image.open(file.stream)
        try:
            stego = embed_message(img, msg)
        except ValueError as e:
            flash(str(e))
            return redirect(request.url)
        buf = io.BytesIO()
        stego.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='steganografi.png', mimetype='image/png')

    # Decode
    if request.method == 'POST' and 'decode' in request.form:
        file = request.files.get('image')
        if not file:
            flash('Gambar wajib diisi!')
            return redirect(request.url)
        img = Image.open(file.stream)
        hidden = extract_message(img)

    return render_template('index.html', hidden=hidden)



if __name__ == '__main__':
    app.run(debug=True)
