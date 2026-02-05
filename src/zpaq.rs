//! Minimal ZPAQ-inspired compressor (context mixing + range coder).
//!
//! This is a small, self-contained subset that keeps the core ideas from
//! libzpaq: bitwise arithmetic coding driven by mixed context predictors.

use std::sync::OnceLock;

const DT2K: [i32; 256] = [
    0, 2048, 1024, 682, 512, 409, 341, 292, 256, 227, 204, 186, 170, 157, 146, 136, 128, 120, 113, 107, 102,
    97, 93, 89, 85, 81, 78, 75, 73, 70, 68, 66, 64, 62, 60, 58, 56, 55, 53, 52, 51, 49, 48, 47, 46, 45, 44,
    43, 42, 41, 40, 40, 39, 38, 37, 37, 36, 35, 35, 34, 34, 33, 33, 32, 32, 31, 31, 30, 30, 29, 29, 28, 28,
    28, 27, 27, 26, 26, 26, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23, 22, 22, 22, 22, 21, 21, 21, 21, 20,
    20, 20, 20, 20, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16,
    16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
];

// State transition table from libzpaq (sns).
const SNS: [u8; 1024] = [
    1, 2, 0, 0, 3, 5, 1, 0, 4, 6, 0, 1, 7, 9, 2, 0, 8, 11, 1, 1, 8, 11, 1, 1, 10, 12, 0, 2, 13, 15, 3, 0, 14,
    17, 2, 1, 14, 17, 2, 1, 16, 19, 1, 2, 16, 19, 1, 2, 18, 20, 0, 3, 21, 23, 4, 0, 22, 25, 3, 1, 22, 25, 3,
    1, 24, 27, 2, 2, 24, 27, 2, 2, 26, 29, 1, 3, 26, 29, 1, 3, 28, 30, 0, 4, 31, 33, 5, 0, 32, 35, 4, 1, 32,
    35, 4, 1, 34, 37, 3, 2, 34, 37, 3, 2, 36, 39, 2, 3, 36, 39, 2, 3, 38, 41, 1, 4, 38, 41, 1, 4, 40, 42, 0,
    5, 43, 33, 6, 0, 44, 47, 5, 1, 44, 47, 5, 1, 46, 49, 4, 2, 46, 49, 4, 2, 48, 51, 3, 3, 48, 51, 3, 3, 50,
    53, 2, 4, 50, 53, 2, 4, 52, 55, 1, 5, 52, 55, 1, 5, 40, 56, 0, 6, 57, 45, 7, 0, 58, 47, 6, 1, 58, 47, 6,
    1, 60, 63, 5, 2, 60, 63, 5, 2, 62, 65, 4, 3, 62, 65, 4, 3, 64, 67, 3, 4, 64, 67, 3, 4, 66, 69, 2, 5, 66,
    69, 2, 5, 52, 71, 1, 6, 52, 71, 1, 6, 54, 72, 0, 7, 73, 59, 8, 0, 74, 61, 7, 1, 74, 61, 7, 1, 76, 63, 6,
    2, 76, 63, 6, 2, 78, 81, 5, 3, 78, 81, 5, 3, 80, 83, 4, 4, 80, 83, 4, 4, 82, 85, 3, 5, 82, 85, 3, 5, 66,
    87, 2, 6, 66, 87, 2, 6, 68, 89, 1, 7, 68, 89, 1, 7, 70, 90, 0, 8, 91, 59, 9, 0, 92, 77, 8, 1, 92, 77, 8,
    1, 94, 79, 7, 2, 94, 79, 7, 2, 96, 81, 6, 3, 96, 81, 6, 3, 98, 101, 5, 4, 98, 101, 5, 4, 100, 103, 4, 5,
    100, 103, 4, 5, 82, 105, 3, 6, 82, 105, 3, 6, 84, 107, 2, 7, 84, 107, 2, 7, 86, 109, 1, 8, 86, 109, 1, 8,
    70, 110, 0, 9, 111, 59, 10, 0, 112, 77, 9, 1, 112, 77, 9, 1, 114, 97, 8, 2, 114, 97, 8, 2, 116, 99, 7, 3,
    116, 99, 7, 3, 62, 101, 6, 4, 62, 101, 6, 4, 80, 83, 5, 5, 80, 83, 5, 5, 100, 67, 4, 6, 100, 67, 4, 6,
    102, 119, 3, 7, 102, 119, 3, 7, 104, 121, 2, 8, 104, 121, 2, 8, 86, 123, 1, 9, 86, 123, 1, 9, 70, 124, 0,
    10, 125, 59, 11, 0, 126, 77, 10, 1, 126, 77, 10, 1, 128, 97, 9, 2, 128, 97, 9, 2, 60, 63, 8, 3, 60, 63,
    8, 3, 66, 69, 3, 8, 66, 69, 3, 8, 104, 131, 2, 9, 104, 131, 2, 9, 86, 133, 1, 10, 86, 133, 1, 10, 70,
    134, 0, 11, 135, 59, 12, 0, 136, 77, 11, 1, 136, 77, 11, 1, 138, 97, 10, 2, 138, 97, 10, 2, 104, 141, 2,
    10, 104, 141, 2, 10, 86, 143, 1, 11, 86, 143, 1, 11, 70, 144, 0, 12, 145, 59, 13, 0, 146, 77, 12, 1, 146,
    77, 12, 1, 148, 97, 11, 2, 148, 97, 11, 2, 104, 151, 2, 11, 104, 151, 2, 11, 86, 153, 1, 12, 86, 153, 1,
    12, 70, 154, 0, 13, 155, 59, 14, 0, 156, 77, 13, 1, 156, 77, 13, 1, 158, 97, 12, 2, 158, 97, 12, 2, 104,
    161, 2, 12, 104, 161, 2, 12, 86, 163, 1, 13, 86, 163, 1, 13, 70, 164, 0, 14, 165, 59, 15, 0, 166, 77, 14,
    1, 166, 77, 14, 1, 168, 97, 13, 2, 168, 97, 13, 2, 104, 171, 2, 13, 104, 171, 2, 13, 86, 173, 1, 14, 86,
    173, 1, 14, 70, 174, 0, 15, 175, 59, 16, 0, 176, 77, 15, 1, 176, 77, 15, 1, 178, 97, 14, 2, 178, 97, 14,
    2, 104, 181, 2, 14, 104, 181, 2, 14, 86, 183, 1, 15, 86, 183, 1, 15, 70, 184, 0, 16, 185, 59, 17, 0, 186,
    77, 16, 1, 186, 77, 16, 1, 74, 97, 15, 2, 74, 97, 15, 2, 104, 89, 2, 15, 104, 89, 2, 15, 86, 187, 1, 16,
    86, 187, 1, 16, 70, 188, 0, 17, 189, 59, 18, 0, 190, 77, 17, 1, 86, 191, 1, 17, 70, 192, 0, 18, 193, 59,
    19, 0, 194, 77, 18, 1, 86, 195, 1, 18, 70, 196, 0, 19, 193, 59, 20, 0, 197, 77, 19, 1, 86, 198, 1, 19,
    70, 196, 0, 20, 199, 77, 20, 1, 86, 200, 1, 20, 201, 77, 21, 1, 86, 202, 1, 21, 203, 77, 22, 1, 86, 204,
    1, 22, 205, 77, 23, 1, 86, 206, 1, 23, 207, 77, 24, 1, 86, 208, 1, 24, 209, 77, 25, 1, 86, 210, 1, 25,
    211, 77, 26, 1, 86, 212, 1, 26, 213, 77, 27, 1, 86, 214, 1, 27, 215, 77, 28, 1, 86, 216, 1, 28, 217, 77,
    29, 1, 86, 218, 1, 29, 219, 77, 30, 1, 86, 220, 1, 30, 221, 77, 31, 1, 86, 222, 1, 31, 223, 77, 32, 1,
    86, 224, 1, 32, 225, 77, 33, 1, 86, 226, 1, 33, 227, 77, 34, 1, 86, 228, 1, 34, 229, 77, 35, 1, 86, 230,
    1, 35, 231, 77, 36, 1, 86, 232, 1, 36, 233, 77, 37, 1, 86, 234, 1, 37, 235, 77, 38, 1, 86, 236, 1, 38,
    237, 77, 39, 1, 86, 238, 1, 39, 239, 77, 40, 1, 86, 240, 1, 40, 241, 77, 41, 1, 86, 242, 1, 41, 243, 77,
    42, 1, 86, 244, 1, 42, 245, 77, 43, 1, 86, 246, 1, 43, 247, 77, 44, 1, 86, 248, 1, 44, 249, 77, 45, 1,
    86, 250, 1, 45, 251, 77, 46, 1, 86, 252, 1, 46, 253, 77, 47, 1, 86, 254, 1, 47, 253, 77, 48, 1, 86, 254,
    1, 48, 0, 0, 0, 0,
];

#[derive(Clone)]
struct Tables {
    squash: Vec<i16>,
    stretch: Vec<i16>,
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut squash = vec![0i16; 4096];
        for i in 0..4096 {
            let x = i as f64 - 2048.0;
            let p = 1.0 / (1.0 + (-x / 64.0).exp());
            let v = (p * 32768.0 + 0.5) as i32;
            squash[i] = v.clamp(0, 32767) as i16;
        }
        let mut stretch = vec![0i16; 32768];
        for p in 0..32768 {
            if p == 0 {
                stretch[p] = -2048;
                continue;
            }
            if p == 32767 {
                stretch[p] = 2047;
                continue;
            }
            let p_f = p as f64;
            let x = ((p_f + 0.5) / (32767.5 - p_f)).ln() * 64.0;
            let xi = x.round().clamp(-2048.0, 2047.0) as i16;
            stretch[p] = xi;
        }
        Tables { squash, stretch }
    })
}

fn squash(x: i32) -> i32 {
    let x = x.clamp(-2048, 2047);
    tables().squash[(x + 2048) as usize] as i32
}

fn stretch(p: i32) -> i32 {
    let p = p.clamp(0, 32767);
    tables().stretch[p as usize] as i32
}

fn clamp2k(x: i32) -> i32 {
    if x < -2048 {
        -2048
    } else if x > 2047 {
        2047
    } else {
        x
    }
}

fn clamp512k(x: i32) -> i32 {
    if x < -524_288 {
        -524_288
    } else if x > 524_287 {
        524_287
    } else {
        x
    }
}

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[derive(Clone)]
struct StateTable {
    ns: [u8; 1024],
}

impl StateTable {
    fn new() -> Self {
        Self { ns: SNS }
    }

    fn next(&self, state: u8, y: u8) -> u8 {
        let idx = state as usize * 4 + y as usize;
        self.ns[idx]
    }

    fn cminit(&self, state: u8) -> i32 {
        let n0 = self.ns[state as usize * 4 + 2] as i32;
        let n1 = self.ns[state as usize * 4 + 3] as i32;
        ((n1 * 2 + 1) << 22) / (n0 + n1 + 1)
    }
}

fn find(ht: &mut [u8], sizebits: u8, cxt: u32) -> usize {
    let size = 16usize << sizebits;
    let chk = ((cxt >> sizebits) & 255) as u8;
    let h0 = (cxt as usize * 16) & (size - 16);
    if ht[h0] == chk {
        return h0;
    }
    let h1 = h0 ^ 16;
    if ht[h1] == chk {
        return h1;
    }
    let h2 = h0 ^ 32;
    if ht[h2] == chk {
        return h2;
    }

    let r = if ht[h0 + 1] <= ht[h1 + 1] && ht[h0 + 1] <= ht[h2 + 1] {
        h0
    } else if ht[h1 + 1] < ht[h2 + 1] {
        h1
    } else {
        h2
    };

    for i in 0..16 {
        ht[r + i] = 0;
    }
    ht[r] = chk;
    r
}

struct Icm {
    sizebits: u8,
    ht: Vec<u8>,
    row: usize,
    state: u8,
}

impl Icm {
    fn new(sizebits: u8) -> Self {
        Self {
            sizebits,
            ht: vec![0; 16usize << sizebits],
            row: 0,
            state: 0,
        }
    }

    fn predict(&mut self, st: &StateTable, h: u32, c8: u32, hmap4: u32) -> i32 {
        if c8 == 1 || (c8 & 0xf0) == 16 {
            let cxt = h.wrapping_add(16 * c8);
            self.row = find(&mut self.ht, self.sizebits, cxt);
        }
        let idx = self.row + (hmap4 as usize & 15);
        self.state = self.ht[idx];
        let p = st.cminit(self.state);
        stretch((p >> 8) as i32)
    }

    fn update(&mut self, st: &StateTable, y: u8, hmap4: u32) {
        let idx = self.row + (hmap4 as usize & 15);
        let next = st.next(self.ht[idx], y);
        self.ht[idx] = next;
    }
}

struct MatchModel {
    cm: Vec<u32>,
    buf: Vec<u8>,
    pos: usize,
    len: u8,
    offset: usize,
    bitpos: u8,
    pred_bit: u8,
    mask: usize,
}

impl MatchModel {
    fn new(sizebits: u8, bufbits: u8) -> Self {
        let cm_size = 1usize << sizebits;
        let buf_size = 1usize << bufbits;
        Self {
            cm: vec![0; cm_size],
            buf: vec![0; buf_size],
            pos: 0,
            len: 0,
            offset: 0,
            bitpos: 0,
            pred_bit: 0,
            mask: buf_size - 1,
        }
    }

    fn predict(&mut self, _ctx: u32) -> i32 {
        if self.len == 0 {
            self.pred_bit = 0;
            return 0;
        }
        let idx = self.pos.wrapping_sub(self.offset) & self.mask;
        let byte = self.buf[idx];
        let bit = (byte >> (7 - self.bitpos)) & 1;
        self.pred_bit = bit;
        let base = DT2K[self.len as usize] as i32;
        let p = if bit == 1 { 32768 - base } else { base };
        stretch(p.clamp(0, 32767))
    }

    fn update(&mut self, ctx: u32, y: u8) {
        if self.len > 0 && self.pred_bit != y {
            self.len = 0;
        }

        let b = self.buf[self.pos];
        self.buf[self.pos] = b.wrapping_add(b).wrapping_add(y);
        self.bitpos += 1;

        if self.bitpos == 8 {
            self.bitpos = 0;
            self.pos = (self.pos + 1) & self.mask;

            if self.len == 0 {
                let idx = (ctx as usize) & (self.cm.len() - 1);
                let prev = self.cm[idx] as usize;
                self.offset = self.pos.wrapping_sub(prev) & self.mask;
                if self.offset != 0 {
                    let mut a: u8 = 0;
                    while a < 255
                        && self.buf[(self.pos + self.mask - a as usize - 1) & self.mask]
                            == self.buf[
                                (self.pos + self.mask - a as usize - 1 - self.offset) & self.mask
                            ]
                    {
                        a = a.wrapping_add(1);
                    }
                    self.len = a;
                }
                self.cm[idx] = self.pos as u32;
            } else if self.len < 255 {
                self.len += 1;
            }
        }
    }
}

struct Mixer {
    weights: Vec<i32>,
    rate: i32,
}

impl Mixer {
    fn new(inputs: usize, rate: i32) -> Self {
        let init = if inputs > 0 { 65536 / inputs as i32 } else { 0 };
        Self {
            weights: vec![init; inputs],
            rate,
        }
    }

    fn mix(&self, inputs: &[i32]) -> i32 {
        let mut sum = 0i32;
        for (w, p) in self.weights.iter().zip(inputs.iter()) {
            sum += (w >> 8) * *p;
        }
        clamp2k(sum >> 8)
    }

    fn update(&mut self, inputs: &[i32], pmix: i32, p1: i32, y: u8) {
        let err = ((y as i32 * 32767 - p1) * self.rate) >> 4;
        for (w, p) in self.weights.iter_mut().zip(inputs.iter()) {
            let delta = (err * *p + (1 << 12)) >> 13;
            *w = clamp512k(*w + delta);
        }
        let _ = pmix; // retained for symmetry with libzpaq
    }
}

struct ZpaqModel {
    st: StateTable,
    icms: Vec<Icm>,
    match_model: MatchModel,
    mixer: Mixer,
    c8: u32,
    hmap4: u32,
    hist: [u8; 4],
    contexts: Vec<u32>,
    match_ctx: u32,
    last_inputs: Vec<i32>,
    last_pmix: i32,
    last_p1: i32,
}

impl ZpaqModel {
    fn new() -> Self {
        let icm_sizes = [19u8, 20u8, 20u8, 21u8, 21u8, 22u8];
        let mut icms = Vec::with_capacity(icm_sizes.len());
        for &s in icm_sizes.iter() {
            icms.push(Icm::new(s));
        }
        let inputs = icms.len() + 1; // + match
        let mut model = Self {
            st: StateTable::new(),
            icms,
            match_model: MatchModel::new(24, 24),
            mixer: Mixer::new(inputs, 3),
            c8: 1,
            hmap4: 1,
            hist: [0; 4],
            contexts: vec![0; icm_sizes.len()],
            match_ctx: 0,
            last_inputs: vec![0; inputs],
            last_pmix: 0,
            last_p1: 2048,
        };
        model.update_contexts(0);
        model
    }

    fn update_contexts(&mut self, byte: u8) {
        self.hist[3] = self.hist[2];
        self.hist[2] = self.hist[1];
        self.hist[1] = self.hist[0];
        self.hist[0] = byte;

        let b1 = self.hist[0] as u32;
        let b2 = self.hist[1] as u32;
        let b3 = self.hist[2] as u32;
        let b4 = self.hist[3] as u32;

        if !self.contexts.is_empty() {
            self.contexts[0] = 0;
        }
        if self.contexts.len() > 1 {
            self.contexts[1] = hash32(b1);
        }
        if self.contexts.len() > 2 {
            self.contexts[2] = hash32(b1 | (b2 << 8));
        }
        if self.contexts.len() > 3 {
            self.contexts[3] = hash32(b1 | (b2 << 8) | (b3 << 16));
        }
        if self.contexts.len() > 4 {
            self.contexts[4] = hash32(b1 | (b2 << 8) | (b3 << 16) | (b4 << 24));
        }
        if self.contexts.len() > 5 {
            self.contexts[5] = hash32(b1 ^ (b2 << 8) ^ (b3 << 16) ^ (b4 << 24));
        }

        self.match_ctx = hash32(b1 | (b2 << 8) | (b3 << 16) | (b4 << 24));
    }

    fn predict(&mut self) -> u32 {
        for i in 0..self.icms.len() {
            let p = self.icms[i].predict(&self.st, self.contexts[i], self.c8, self.hmap4);
            self.last_inputs[i] = p;
        }
        let match_p = self.match_model.predict(self.match_ctx);
        self.last_inputs[self.icms.len()] = match_p;

        self.last_pmix = self.mixer.mix(&self.last_inputs);
        self.last_p1 = squash(self.last_pmix);
        self.last_p1 as u32
    }

    fn update(&mut self, y: u8) {
        for icm in self.icms.iter_mut() {
            icm.update(&self.st, y, self.hmap4);
        }
        self.match_model.update(self.match_ctx, y);
        self.mixer
            .update(&self.last_inputs, self.last_pmix, self.last_p1, y);

        self.c8 = (self.c8 << 1) | y as u32;
        if self.c8 >= 256 {
            let byte = (self.c8 - 256) as u8;
            self.c8 = 1;
            self.hmap4 = 1;
            self.update_contexts(byte);
        } else if self.c8 >= 16 && self.c8 < 32 {
            self.hmap4 = ((self.hmap4 & 0xF) << 5) | ((y as u32) << 4) | 1;
        } else {
            self.hmap4 = (self.hmap4 & 0x1F0) | (((self.hmap4 & 0xF) * 2 + y as u32) & 0xF);
        }
    }
}

struct Encoder {
    low: u32,
    high: u32,
    out: Vec<u8>,
    model: ZpaqModel,
}

impl Encoder {
    fn new() -> Self {
        Self {
            low: 1,
            high: 0xFFFF_FFFF,
            out: Vec::new(),
            model: ZpaqModel::new(),
        }
    }

    fn encode_bit(&mut self, y: u8, p: u32) {
        let range = (self.high - self.low) as u64;
        let mid = self.low + ((range * p as u64) >> 16) as u32;
        if y == 1 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }
        while (self.high ^ self.low) < 0x0100_0000 {
            self.out.push((self.high >> 24) as u8);
            self.high = (self.high << 8) | 0xFF;
            self.low <<= 8;
            if self.low == 0 {
                self.low = 1;
            }
        }
    }

    fn compress_byte(&mut self, byte: u8) {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u8;
            let p1 = self.model.predict();
            let p = p1 * 2 + 1; // 1..65535
            self.encode_bit(bit, p);
            self.model.update(bit);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Emit 4 bytes to finalize the range.
        self.out.push((self.low >> 24) as u8);
        self.out.push((self.low >> 16) as u8);
        self.out.push((self.low >> 8) as u8);
        self.out.push(self.low as u8);
        self.out
    }
}

struct Decoder<'a> {
    low: u32,
    high: u32,
    curr: u32,
    input: &'a [u8],
    pos: usize,
    model: ZpaqModel,
}

impl<'a> Decoder<'a> {
    fn new(input: &'a [u8]) -> Self {
        let mut dec = Self {
            low: 1,
            high: 0xFFFF_FFFF,
            curr: 0,
            input,
            pos: 0,
            model: ZpaqModel::new(),
        };
        for _ in 0..4 {
            dec.curr = (dec.curr << 8) | dec.get_byte() as u32;
        }
        dec
    }

    fn get_byte(&mut self) -> u8 {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    fn decode_bit(&mut self, p: u32) -> u8 {
        let range = (self.high - self.low) as u64;
        let mid = self.low + ((range * p as u64) >> 16) as u32;
        let y;
        if self.curr <= mid {
            y = 1;
            self.high = mid;
        } else {
            y = 0;
            self.low = mid + 1;
        }
        while (self.high ^ self.low) < 0x0100_0000 {
            self.high = (self.high << 8) | 0xFF;
            self.low <<= 8;
            if self.low == 0 {
                self.low = 1;
            }
            self.curr = (self.curr << 8) | self.get_byte() as u32;
        }
        y
    }

    fn decompress_byte(&mut self) -> u8 {
        let mut c: u8 = 0;
        for _ in 0..8 {
            let p1 = self.model.predict();
            let p = p1 * 2 + 1;
            let bit = self.decode_bit(p);
            self.model.update(bit);
            c = (c << 1) | bit;
        }
        c
    }
}

pub fn compress_text(data: &[u8]) -> Vec<u8> {
    let mut encoder = Encoder::new();
    for &b in data {
        encoder.compress_byte(b);
    }
    let mut payload = encoder.finish();

    let mut out = Vec::with_capacity(payload.len() + 4);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.append(&mut payload);
    out
}

pub fn decompress_text(data: &[u8]) -> Vec<u8> {
    if data.len() < 4 {
        return Vec::new();
    }
    let size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let payload = &data[4..];

    let mut decoder = Decoder::new(payload);
    let mut out = Vec::with_capacity(size);
    for _ in 0..size {
        out.push(decoder.decompress_byte());
    }
    out
}
