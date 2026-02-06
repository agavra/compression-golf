//! ZPAQ core compressor/decompressor (in-memory), ported from libzpaq.
//!
//! This implements the ZPAQ block format and context-mixing arithmetic coder
//! using the built-in max model (level 3) headers from libzpaq.

use crate::zpaq_consts::*;

use std::cmp::{max, min};

pub fn compress_text(data: &[u8]) -> Vec<u8> {
    let mut out = ByteWriter::new();

    // Load built-in max model header
    let mut z = ZPAQL::new();
    let mut hdr_reader = ByteReader::new(MODEL_MAX);
    z.read(&mut hdr_reader);

    // Write ZPAQ block header
    out.put(b'z' as i32);
    out.put(b'P' as i32);
    out.put(b'Q' as i32);
    out.put(if z.header[6] == 0 { 1 } else { 2 });
    out.put(1);
    z.write(&mut out, false);

    // Segment header (empty filename/comment)
    out.put(1);
    out.put(0);
    out.put(0);
    out.put(0);

    // Encode (no PCOMP, so PASS=0)
    let mut enc = Encoder::new(z.clone());
    enc.init();
    enc.compress(&mut out, 0);

    for &b in data {
        enc.compress(&mut out, b as i32);
    }
    enc.compress(&mut out, -1);

    // Segment end marker
    out.put(0);
    out.put(0);
    out.put(0);
    out.put(0);
    out.put(254);

    out.into_inner()
}

pub fn decompress_text(data: &[u8]) -> Vec<u8> {
    let mut reader = ByteReader::new(data);

    // Read tag
    let mut tag = [0u8; 4];
    for i in 0..4 {
        let c = reader.get();
        if c < 0 {
            error("unexpected end of file");
        }
        tag[i] = c as u8;
    }
    if &tag[0..3] != b"zPQ" {
        error("bad ZPAQ tag");
    }
    let _ = tag[3];
    let _ = reader.get(); // version

    let mut z = ZPAQL::new();
    z.read(&mut reader);

    // Segment header
    if reader.get() != 1 {
        error("missing segment");
    }
    while reader.get() != 0 {}
    while reader.get() != 0 {}
    if reader.get() != 0 {
        error("bad segment header");
    }

    // Decode stream (no PCOMP, so PASS)
    let mut dec = Decoder::new(z);
    dec.init();
    dec.load(&mut reader);

    let mut out = Vec::new();
    loop {
        let ch = dec.decompress(&mut reader);
        if ch < 0 {
            break;
        }
        out.push(ch as u8);
    }

    // Skip end marker
    for _ in 0..4 {
        let _ = reader.get();
    }
    let _ = reader.get();

    out
}

#[derive(Debug)]
struct ZpaqError(&'static str);

fn error(msg: &'static str) -> ! {
    panic!("ZPAQ error: {msg}")
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn get(&mut self) -> i32 {
        if self.pos >= self.data.len() {
            return -1;
        }
        let c = self.data[self.pos];
        self.pos += 1;
        c as i32
    }
}

struct ByteWriter {
    data: Vec<u8>,
}

impl ByteWriter {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn put(&mut self, c: i32) {
        self.data.push(c as u8);
    }

    fn write(&mut self, buf: &[u8]) {
        self.data.extend_from_slice(buf);
    }

    fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

#[derive(Clone)]
struct Array<T: Default + Clone> {
    data: Vec<T>,
}

impl<T: Default + Clone> Array<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn resize(&mut self, sz: usize, ex: usize) {
        let mut size = sz;
        for _ in 0..ex {
            size = size.checked_mul(2).unwrap_or_else(|| error("Array too big"));
        }
        self.data = vec![T::default(); size];
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn isize(&self) -> usize {
        self.data.len()
    }

    fn at(&self, i: usize) -> &T {
        &self.data[i]
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    fn at_mod(&self, i: usize) -> &T {
        let mask = self.data.len() - 1;
        &self.data[i & mask]
    }

    fn at_mod_mut(&mut self, i: usize) -> &mut T {
        let mask = self.data.len() - 1;
        &mut self.data[i & mask]
    }
}

impl<T: Default + Clone> Default for Array<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Default)]
struct Component {
    limit: usize,
    cxt: usize,
    a: usize,
    b: usize,
    c: usize,
    cm: Array<u32>,
    ht: Array<u8>,
    a16: Array<u16>,
}

impl Component {
    fn init(&mut self) {
        self.limit = 0;
        self.cxt = 0;
        self.a = 0;
        self.b = 0;
        self.c = 0;
        self.cm = Array::new();
        self.ht = Array::new();
        self.a16 = Array::new();
    }
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
        self.ns[state as usize * 4 + y as usize]
    }

    fn cminit(&self, state: u8) -> u32 {
        let n0 = self.ns[state as usize * 4 + 2] as u32;
        let n1 = self.ns[state as usize * 4 + 3] as u32;
        ((n1 * 2 + 1) << 22) / (n0 + n1 + 1)
    }
}

#[derive(Clone)]
struct ZPAQL {
    header: Vec<u8>,
    cend: usize,
    hbegin: usize,
    hend: usize,
    m: Array<u8>,
    h: Array<u32>,
    r: Array<u32>,
    outbuf: Vec<u8>,
    bufptr: usize,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    f: i32,
    pc: usize,
}

impl ZPAQL {
    fn new() -> Self {
        Self {
            header: Vec::new(),
            cend: 0,
            hbegin: 0,
            hend: 0,
            m: Array::new(),
            h: Array::new(),
            r: Array::new(),
            outbuf: vec![0u8; 1 << 14],
            bufptr: 0,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            f: 0,
            pc: 0,
        }
    }

    fn clear(&mut self) {
        self.header.clear();
        self.cend = 0;
        self.hbegin = 0;
        self.hend = 0;
        self.m = Array::new();
        self.h = Array::new();
        self.r = Array::new();
        self.a = 0;
        self.b = 0;
        self.c = 0;
        self.d = 0;
        self.f = 0;
        self.pc = 0;
    }

    fn inith(&mut self) {
        if self.header.len() <= 6 {
            error("Missing header");
        }
        self.init(self.header[2] as usize, self.header[3] as usize);
    }

    fn initp(&mut self) {
        if self.header.len() <= 6 {
            error("Missing header");
        }
        self.init(self.header[4] as usize, self.header[5] as usize);
    }

    fn init(&mut self, hbits: usize, mbits: usize) {
        if hbits > 32 || mbits > 32 {
            error("H or M too big");
        }
        self.h.resize(1, hbits);
        self.m.resize(1, mbits);
        self.r.resize(256, 0);
        self.a = 0;
        self.b = 0;
        self.c = 0;
        self.d = 0;
        self.f = 0;
        self.pc = 0;
    }

    fn h_at(&mut self, idx: u32) -> &mut u32 {
        self.h.at_mod_mut(idx as usize)
    }

    fn m_at(&mut self, idx: u32) -> &mut u8 {
        self.m.at_mod_mut(idx as usize)
    }

    fn outc(&mut self, ch: i32) {
        if ch < 0 {
            return;
        }
        if self.bufptr >= self.outbuf.len() {
            self.bufptr = 0;
        }
        self.outbuf[self.bufptr] = ch as u8;
        self.bufptr += 1;
    }

    fn take_output(&mut self, out: &mut Vec<u8>) {
        if self.bufptr > 0 {
            out.extend_from_slice(&self.outbuf[..self.bufptr]);
            self.bufptr = 0;
        }
    }

    fn read(&mut self, input: &mut ByteReader) {
        let hsize0 = input.get();
        let hsize1 = input.get();
        if hsize0 < 0 || hsize1 < 0 {
            error("unexpected end of file");
        }
        let hsize = hsize0 as usize + ((hsize1 as usize) << 8);
        self.header = vec![0u8; hsize + 300];
        self.cend = 0;
        self.hbegin = 0;
        self.hend = 0;
        self.header[self.cend] = hsize0 as u8;
        self.cend += 1;
        self.header[self.cend] = hsize1 as u8;
        self.cend += 1;
        while self.cend < 7 {
            let c = input.get();
            if c < 0 {
                error("unexpected end of file");
            }
            self.header[self.cend] = c as u8;
            self.cend += 1;
        }
        let n = self.header[self.cend - 1] as usize;
        for _ in 0..n {
            let typ = input.get();
            if typ < 0 {
                error("unexpected end of file");
            }
            let typ = typ as u8;
            self.header[self.cend] = typ;
            self.cend += 1;
            let size = COMPSIZE[typ as usize];
            if size < 1 {
                error("Invalid component type");
            }
            for _ in 1..size {
                let c = input.get();
                if c < 0 {
                    error("unexpected end of file");
                }
                self.header[self.cend] = c as u8;
                self.cend += 1;
            }
        }
        let end = input.get();
        if end != 0 {
            error("missing COMP END");
        }
        self.header[self.cend] = 0;
        self.cend += 1;

        self.hbegin = self.cend + 128;
        self.hend = self.hbegin;
        while self.hend < hsize + 129 {
            let op = input.get();
            if op < 0 {
                error("unexpected end of file");
            }
            self.header[self.hend] = op as u8;
            self.hend += 1;
        }
    }

    fn write(&self, out: &mut ByteWriter, pp: bool) {
        if self.header.len() <= 6 {
            return;
        }
        if !pp {
            for i in 0..self.cend {
                out.put(self.header[i] as i32);
            }
        } else {
            let len = self.hend - self.hbegin;
            out.put((len & 255) as i32);
            out.put((len >> 8) as i32);
        }
        for i in self.hbegin..self.hend {
            out.put(self.header[i] as i32);
        }
    }

    fn run(&mut self, input: u32, out: &mut Vec<u8>) {
        self.pc = self.hbegin;
        self.a = input;
        while self.execute() {}
        self.take_output(out);
    }

    fn execute(&mut self) -> bool {
        let op = self.header[self.pc];
        self.pc += 1;
        match op {
            0 => error("ZPAQL execution error"),
            1 => self.a = self.a.wrapping_add(1),
            2 => self.a = self.a.wrapping_sub(1),
            3 => self.a = !self.a,
            4 => self.a = 0,
            7 => {
                let idx = self.header[self.pc] as usize;
                self.pc += 1;
                self.a = *self.r.at(idx);
            }
            8 => {
                let b = self.b;
                self.b = self.a;
                self.a = b;
            }
            9 => self.b = self.b.wrapping_add(1),
            10 => self.b = self.b.wrapping_sub(1),
            11 => self.b = !self.b,
            12 => self.b = 0,
            15 => {
                let idx = self.header[self.pc] as usize;
                self.pc += 1;
                self.b = *self.r.at(idx);
            }
            16 => {
                let c = self.c;
                self.c = self.a;
                self.a = c;
            }
            17 => self.c = self.c.wrapping_add(1),
            18 => self.c = self.c.wrapping_sub(1),
            19 => self.c = !self.c,
            20 => self.c = 0,
            23 => {
                let idx = self.header[self.pc] as usize;
                self.pc += 1;
                self.c = *self.r.at(idx);
            }
            24 => {
                let d = self.d;
                self.d = self.a;
                self.a = d;
            }
            25 => self.d = self.d.wrapping_add(1),
            26 => self.d = self.d.wrapping_sub(1),
            27 => self.d = !self.d,
            28 => self.d = 0,
            31 => {
                let idx = self.header[self.pc] as usize;
                self.pc += 1;
                self.d = *self.r.at(idx);
            }
            32 => {
                let b = self.b as usize;
                let mval = *self.m.at_mod(b) as u32;
                self.a ^= mval;
                let new_m = (mval ^ self.a) as u8;
                *self.m.at_mod_mut(b) = new_m;
                self.a ^= new_m as u32;
            }
            33 => {
                let b = self.b;
                let m = self.m_at(b);
                *m = m.wrapping_add(1);
            }
            34 => {
                let b = self.b;
                let m = self.m_at(b);
                *m = m.wrapping_sub(1);
            }
            35 => {
                let b = self.b;
                let m = self.m_at(b);
                *m = !*m;
            }
            36 => {
                let b = self.b;
                let m = self.m_at(b);
                *m = 0;
            }
            39 => {
                let off = self.header[self.pc] as i32;
                self.pc += 1;
                if self.f != 0 {
                    self.pc = ((self.pc as i32) + ((off + 128) & 255) - 127) as usize;
                }
            }
            40 => {
                let c = self.c as usize;
                let mval = *self.m.at_mod(c) as u32;
                self.a ^= mval;
                let new_m = (mval ^ self.a) as u8;
                *self.m.at_mod_mut(c) = new_m;
                self.a ^= new_m as u32;
            }
            41 => {
                let c = self.c;
                let m = self.m_at(c);
                *m = m.wrapping_add(1);
            }
            42 => {
                let c = self.c;
                let m = self.m_at(c);
                *m = m.wrapping_sub(1);
            }
            43 => {
                let c = self.c;
                let m = self.m_at(c);
                *m = !*m;
            }
            44 => {
                let c = self.c;
                let m = self.m_at(c);
                *m = 0;
            }
            47 => {
                let off = self.header[self.pc] as i32;
                self.pc += 1;
                if self.f == 0 {
                    self.pc = ((self.pc as i32) + ((off + 128) & 255) - 127) as usize;
                }
            }
            48 => {
                let d = self.d as usize;
                let hval = *self.h.at_mod(d);
                self.a ^= hval;
                let new_h = hval ^ self.a;
                *self.h.at_mod_mut(d) = new_h;
                self.a ^= new_h;
            }
            49 => {
                let d = self.d;
                let h = self.h_at(d);
                *h = h.wrapping_add(1);
            }
            50 => {
                let d = self.d;
                let h = self.h_at(d);
                *h = h.wrapping_sub(1);
            }
            51 => {
                let d = self.d;
                let h = self.h_at(d);
                *h = !*h;
            }
            52 => {
                let d = self.d;
                let h = self.h_at(d);
                *h = 0;
            }
            55 => {
                let idx = self.header[self.pc] as usize;
                self.pc += 1;
                *self.r.at_mut(idx) = self.a;
            }
            56 => return false,
            57 => self.outc((self.a & 255) as i32),
            59 => {
                let b = self.b;
                let mb = *self.m_at(b) as u32;
                self.a = (self.a.wrapping_add(mb).wrapping_add(512)).wrapping_mul(773);
            }
            60 => {
                let d = self.d;
                let a = self.a;
                let hd = self.h_at(d);
                *hd = hd.wrapping_add(a).wrapping_add(512).wrapping_mul(773);
            }
            63 => {
                let off = self.header[self.pc] as i32;
                self.pc += 1;
                self.pc = ((self.pc as i32) + ((off + 128) & 255) - 127) as usize;
            }
            64 => {}
            65 => self.a = self.b,
            66 => self.a = self.c,
            67 => self.a = self.d,
            68 => self.a = *self.m_at(self.b) as u32,
            69 => self.a = *self.m_at(self.c) as u32,
            70 => self.a = *self.h_at(self.d),
            71 => {
                self.a = self.header[self.pc] as u32;
                self.pc += 1;
            }
            72 => self.b = self.a,
            73 => {}
            74 => self.b = self.c,
            75 => self.b = self.d,
            76 => self.b = *self.m_at(self.b) as u32,
            77 => self.b = *self.m_at(self.c) as u32,
            78 => self.b = *self.h_at(self.d),
            79 => {
                self.b = self.header[self.pc] as u32;
                self.pc += 1;
            }
            80 => self.c = self.a,
            81 => self.c = self.b,
            82 => {}
            83 => self.c = self.d,
            84 => self.c = *self.m_at(self.b) as u32,
            85 => self.c = *self.m_at(self.c) as u32,
            86 => self.c = *self.h_at(self.d),
            87 => {
                self.c = self.header[self.pc] as u32;
                self.pc += 1;
            }
            88 => self.d = self.a,
            89 => self.d = self.b,
            90 => self.d = self.c,
            91 => {}
            92 => self.d = *self.m_at(self.b) as u32,
            93 => self.d = *self.m_at(self.c) as u32,
            94 => self.d = *self.h_at(self.d),
            95 => {
                self.d = self.header[self.pc] as u32;
                self.pc += 1;
            }
            96 => *self.m_at(self.b) = self.a as u8,
            97 => *self.m_at(self.b) = self.b as u8,
            98 => *self.m_at(self.b) = self.c as u8,
            99 => *self.m_at(self.b) = self.d as u8,
            100 => {
                let v = *self.m_at(self.b);
                *self.m_at(self.b) = v;
            }
            101 => *self.m_at(self.b) = *self.m_at(self.c),
            102 => *self.m_at(self.b) = (*self.h_at(self.d) & 255) as u8,
            103 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.m_at(self.b) = v;
            }
            104 => *self.m_at(self.c) = self.a as u8,
            105 => *self.m_at(self.c) = self.b as u8,
            106 => *self.m_at(self.c) = self.c as u8,
            107 => *self.m_at(self.c) = self.d as u8,
            108 => *self.m_at(self.c) = *self.m_at(self.b),
            109 => {
                let v = *self.m_at(self.c);
                *self.m_at(self.c) = v;
            }
            110 => *self.m_at(self.c) = (*self.h_at(self.d) & 255) as u8,
            111 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.m_at(self.c) = v;
            }
            112 => *self.h_at(self.d) = self.a,
            113 => *self.h_at(self.d) = self.b,
            114 => *self.h_at(self.d) = self.c,
            115 => *self.h_at(self.d) = self.d,
            116 => *self.h_at(self.d) = *self.m_at(self.b) as u32,
            117 => *self.h_at(self.d) = *self.m_at(self.c) as u32,
            118 => {
                let v = *self.h_at(self.d);
                *self.h_at(self.d) = v;
            }
            119 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.h_at(self.d) = v as u32;
            }
            120 => {
                *self.m_at(self.b) = self.a as u8;
                self.b = self.a;
            }
            121 => {
                *self.m_at(self.b) = self.b as u8;
                self.b = self.a;
            }
            122 => {
                *self.m_at(self.b) = self.c as u8;
                self.b = self.a;
            }
            123 => {
                *self.m_at(self.b) = self.d as u8;
                self.b = self.a;
            }
            124 => {
                *self.m_at(self.b) = *self.m_at(self.b);
                self.b = self.a;
            }
            125 => {
                *self.m_at(self.b) = *self.m_at(self.c);
                self.b = self.a;
            }
            126 => {
                *self.m_at(self.b) = (*self.h_at(self.d) & 255) as u8;
                self.b = self.a;
            }
            127 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.m_at(self.b) = v;
                self.b = self.a;
            }
            128 => {
                *self.m_at(self.c) = self.a as u8;
                self.c = self.a;
            }
            129 => {
                *self.m_at(self.c) = self.b as u8;
                self.c = self.a;
            }
            130 => {
                *self.m_at(self.c) = self.c as u8;
                self.c = self.a;
            }
            131 => {
                *self.m_at(self.c) = self.d as u8;
                self.c = self.a;
            }
            132 => {
                *self.m_at(self.c) = *self.m_at(self.b);
                self.c = self.a;
            }
            133 => {
                *self.m_at(self.c) = *self.m_at(self.c);
                self.c = self.a;
            }
            134 => {
                *self.m_at(self.c) = (*self.h_at(self.d) & 255) as u8;
                self.c = self.a;
            }
            135 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.m_at(self.c) = v;
                self.c = self.a;
            }
            136 => {
                *self.h_at(self.d) = self.a;
                self.d = self.a;
            }
            137 => {
                *self.h_at(self.d) = self.b;
                self.d = self.a;
            }
            138 => {
                *self.h_at(self.d) = self.c;
                self.d = self.a;
            }
            139 => {
                *self.h_at(self.d) = self.d;
                self.d = self.a;
            }
            140 => {
                *self.h_at(self.d) = *self.m_at(self.b) as u32;
                self.d = self.a;
            }
            141 => {
                *self.h_at(self.d) = *self.m_at(self.c) as u32;
                self.d = self.a;
            }
            142 => {
                *self.h_at(self.d) = *self.h_at(self.d);
                self.d = self.a;
            }
            143 => {
                let v = self.header[self.pc];
                self.pc += 1;
                *self.h_at(self.d) = v as u32;
                self.d = self.a;
            }
            160 => self.a = self.a.wrapping_add(self.a),
            161 => self.a = self.a.wrapping_add(self.b),
            162 => self.a = self.a.wrapping_add(self.c),
            163 => self.a = self.a.wrapping_add(self.d),
            164 => self.a = self.a.wrapping_add(*self.m_at(self.b) as u32),
            165 => self.a = self.a.wrapping_add(*self.m_at(self.c) as u32),
            166 => self.a = self.a.wrapping_add(*self.h_at(self.d)),
            167 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_add(v);
            }
            168 => self.a = self.a.wrapping_sub(self.a),
            169 => self.a = self.a.wrapping_sub(self.b),
            170 => self.a = self.a.wrapping_sub(self.c),
            171 => self.a = self.a.wrapping_sub(self.d),
            172 => self.a = self.a.wrapping_sub(*self.m_at(self.b) as u32),
            173 => self.a = self.a.wrapping_sub(*self.m_at(self.c) as u32),
            174 => self.a = self.a.wrapping_sub(*self.h_at(self.d)),
            175 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_sub(v);
            }
            176 => self.a = self.a.wrapping_mul(self.a),
            177 => self.a = self.a.wrapping_mul(self.b),
            178 => self.a = self.a.wrapping_mul(self.c),
            179 => self.a = self.a.wrapping_mul(self.d),
            180 => self.a = self.a.wrapping_mul(*self.m_at(self.b) as u32),
            181 => self.a = self.a.wrapping_mul(*self.m_at(self.c) as u32),
            182 => self.a = self.a.wrapping_mul(*self.h_at(self.d)),
            183 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_mul(v);
            }
            184 => {
                let v = self.a;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            185 => {
                let v = self.b;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            186 => {
                let v = self.c;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            187 => {
                let v = self.d;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            188 => {
                let v = *self.m_at(self.b) as u32;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            189 => {
                let v = *self.m_at(self.c) as u32;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            190 => {
                let v = *self.h_at(self.d);
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            191 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            192 => {
                let v = self.a;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            193 => {
                let v = self.b;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            194 => {
                let v = self.c;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            195 => {
                let v = self.d;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            196 => {
                let v = *self.m_at(self.b) as u32;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            197 => {
                let v = *self.m_at(self.c) as u32;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            198 => {
                let v = *self.h_at(self.d);
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            199 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            200 => self.a <<= self.a & 31,
            201 => self.a <<= self.b & 31,
            202 => self.a <<= self.c & 31,
            203 => self.a <<= self.d & 31,
            204 => self.a <<= (*self.m_at(self.b) as u32) & 31,
            205 => self.a <<= (*self.m_at(self.c) as u32) & 31,
            206 => self.a <<= (*self.h_at(self.d)) & 31,
            207 => {
                let v = self.header[self.pc] as u32 & 31;
                self.pc += 1;
                self.a <<= v;
            }
            208 => self.a >>= self.a & 31,
            209 => self.a >>= self.b & 31,
            210 => self.a >>= self.c & 31,
            211 => self.a >>= self.d & 31,
            212 => self.a >>= (*self.m_at(self.b) as u32) & 31,
            213 => self.a >>= (*self.m_at(self.c) as u32) & 31,
            214 => self.a >>= (*self.h_at(self.d)) & 31,
            215 => {
                let v = self.header[self.pc] as u32 & 31;
                self.pc += 1;
                self.a >>= v;
            }
            216 => self.f = 1,
            217 => self.f = if self.a == self.b { 1 } else { 0 },
            218 => self.f = if self.a == self.c { 1 } else { 0 },
            219 => self.f = if self.a == self.d { 1 } else { 0 },
            220 => self.f = if self.a == *self.m_at(self.b) as u32 { 1 } else { 0 },
            221 => self.f = if self.a == *self.m_at(self.c) as u32 { 1 } else { 0 },
            222 => self.f = if self.a == *self.h_at(self.d) { 1 } else { 0 },
            223 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.f = if self.a == v { 1 } else { 0 };
            }
            224 => self.f = 0,
            225 => self.f = if self.a < self.b { 1 } else { 0 },
            226 => self.f = if self.a < self.c { 1 } else { 0 },
            227 => self.f = if self.a < self.d { 1 } else { 0 },
            228 => self.f = if self.a < *self.m_at(self.b) as u32 { 1 } else { 0 },
            229 => self.f = if self.a < *self.m_at(self.c) as u32 { 1 } else { 0 },
            230 => self.f = if self.a < *self.h_at(self.d) { 1 } else { 0 },
            231 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.f = if self.a < v { 1 } else { 0 };
            }
            232 => self.f = 0,
            233 => self.f = if self.a > self.b { 1 } else { 0 },
            234 => self.f = if self.a > self.c { 1 } else { 0 },
            235 => self.f = if self.a > self.d { 1 } else { 0 },
            236 => self.f = if self.a > *self.m_at(self.b) as u32 { 1 } else { 0 },
            237 => self.f = if self.a > *self.m_at(self.c) as u32 { 1 } else { 0 },
            238 => self.f = if self.a > *self.h_at(self.d) { 1 } else { 0 },
            239 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.f = if self.a > v { 1 } else { 0 };
            }
            255 => {
                let lo = self.header[self.pc] as usize;
                let hi = self.header[self.pc + 1] as usize;
                self.pc = self.hbegin + lo + (hi << 8);
                if self.pc >= self.hend {
                    error("ZPAQL execution error");
                }
            }
            _ => error("ZPAQL execution error"),
        }
        true
    }

    fn h(&self, i: usize) -> u32 {
        *self.h.at(i)
    }
}

struct Predictor {
    c8: i32,
    hmap4: i32,
    p: [i32; 256],
    h: [u32; 256],
    z: ZPAQL,
    comp: [Component; 256],
    init_tables: bool,
    dt2k: [i32; 256],
    dt: [i32; 1024],
    squasht: [u16; 4096],
    stretcht: [i16; 32768],
    st: StateTable,
}

impl Predictor {
    fn new(z: ZPAQL) -> Self {
        Self {
            c8: 1,
            hmap4: 1,
            p: [0; 256],
            h: [0; 256],
            z,
            comp: std::array::from_fn(|_| Component::default()),
            init_tables: false,
            dt2k: [0; 256],
            dt: [0; 1024],
            squasht: [0; 4096],
            stretcht: [0; 32768],
            st: StateTable::new(),
        }
    }

    fn init(&mut self) {
        if !self.init_tables {
            self.init_tables = true;
            for i in 0..256 {
                self.dt2k[i] = SDT2K[i];
            }
            for i in 0..1024 {
                self.dt[i] = SDT[i];
            }
            for i in 0..1376 {
                self.squasht[i] = 0;
            }
            for i in 0..SSQUASHT.len() {
                self.squasht[1376 + i] = SSQUASHT[i];
            }
            for i in 2720..4096 {
                self.squasht[i] = 32767;
            }
            let mut k = 16384usize;
            for i in 0..STDT.len() {
                for _ in 0..STDT[i] {
                    self.stretcht[k] = i as i16;
                    k += 1;
                }
            }
            for i in 0..16384 {
                self.stretcht[i] = -self.stretcht[32767 - i];
            }
        }

        // Initialize ZPAQL HCOMP memory/state before predictor uses it.
        self.z.inith();

        for i in 0..256 {
            self.h[i] = 0;
            self.p[i] = 0;
            self.comp[i].init();
        }

        let n = self.z.header[6] as usize;
        let mut cp = 7usize;
        for i in 0..n {
            let typ = self.z.header[cp];
            let cr = &mut self.comp[i];
            match typ {
                1 => {
                    self.p[i] = ((self.z.header[cp + 1] as i32) - 128) * 4;
                }
                2 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    let limit = self.z.header[cp + 2] as usize * 4;
                    cr.cm.resize(1, sizebits);
                    cr.limit = limit;
                    for j in 0..cr.cm.size() {
                        *cr.cm.at_mut(j) = 0x8000_0000;
                    }
                }
                3 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    cr.limit = 1023;
                    cr.cm.resize(256, 0);
                    cr.ht.resize(64, sizebits);
                    for j in 0..cr.cm.size() {
                        *cr.cm.at_mut(j) = self.st.cminit(j as u8);
                    }
                }
                4 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    let bufbits = self.z.header[cp + 2] as usize;
                    cr.cm.resize(1, sizebits);
                    cr.ht.resize(1, bufbits);
                    *cr.ht.at_mut(0) = 1;
                }
                5 => {}
                6 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    cr.c = 1usize << sizebits;
                    cr.a16.resize(1, sizebits);
                    for j in 0..cr.a16.size() {
                        *cr.a16.at_mut(j) = 32768;
                    }
                }
                7 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    let m = self.z.header[cp + 3] as usize;
                    cr.c = 1usize << sizebits;
                    cr.cm.resize(m, sizebits);
                    for j in 0..cr.cm.size() {
                        *cr.cm.at_mut(j) = (65536 / max(1, m) as u32) as u32;
                    }
                }
                8 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    cr.ht.resize(64, sizebits);
                    cr.cm.resize(512, 0);
                    for j in 0..256 {
                        *cr.cm.at_mut(j * 2) = 1 << 15;
                        let cminit = self.st.cminit(j as u8);
                        let stretch = stretch((cminit >> 8) as i32, &self.stretcht);
                        let v = clamp512k(stretch * 1024);
                        *cr.cm.at_mut(j * 2 + 1) = v as u32;
                    }
                }
                9 => {
                    let sizebits = self.z.header[cp + 1] as usize;
                    let start = self.z.header[cp + 3] as u32;
                    let limit = self.z.header[cp + 4] as u32 * 4;
                    cr.cm.resize(32, sizebits);
                    cr.limit = limit as usize;
                    for j in 0..cr.cm.size() {
                        let p = squash(((j & 31) as i32) * 64 - 992, &self.squasht);
                        *cr.cm.at_mut(j) = ((p as u32) << 17) | start;
                    }
                }
                _ => {}
            }
            cp += COMPSIZE[typ as usize] as usize;
        }
    }

    fn is_modeled(&self) -> bool {
        self.z.header.len() > 6 && self.z.header[6] != 0
    }

    fn predict(&mut self) -> i32 {
        self.predict0()
    }

    fn update(&mut self, y: i32) {
        self.update0(y)
    }

    fn predict0(&mut self) -> i32 {
        let n = self.z.header[6] as usize;
        let mut cp = 7usize;
        for i in 0..n {
            let typ = self.z.header[cp];
            let cr = &mut self.comp[i];
            match typ {
                1 => {}
                2 => {
                    cr.cxt = (self.h[i] ^ (self.hmap4 as u32)) as usize;
                    let v = *cr.cm.at_mod(cr.cxt);
                    self.p[i] = stretch((v >> 17) as i32, &self.stretcht);
                }
                3 => {
                    if self.c8 == 1 || (self.c8 & 0xf0) == 16 {
                        cr.c = find(&mut cr.ht, (self.z.header[cp + 1] + 2) as usize, self.h[i] + 16 * self.c8 as u32);
                    }
                    cr.cxt = *cr.ht.at(cr.c + (self.hmap4 as usize & 15)) as usize;
                    let v = *cr.cm.at(cr.cxt);
                    self.p[i] = stretch((v >> 8) as i32, &self.stretcht);
                }
                4 => {
                    if cr.a == 0 {
                        self.p[i] = 0;
                    } else {
                        let idx = cr.limit.wrapping_sub(cr.b) & (cr.ht.size() - 1);
                        cr.c = ((*(cr.ht.at(idx)) >> (7 - cr.cxt)) & 1) as usize;
                        let val = self.dt2k[cr.a] * (cr.c as i32 * -2 + 1);
                        self.p[i] = stretch((val & 32767) as i32, &self.stretcht);
                    }
                }
                5 => {
                    let j = self.z.header[cp + 1] as usize;
                    let k = self.z.header[cp + 2] as usize;
                    let wt = self.z.header[cp + 3] as i32;
                    self.p[i] = (self.p[j] * wt + self.p[k] * (256 - wt)) >> 8;
                }
                6 => {
                    let mask = self.z.header[cp + 5] as i32;
                    cr.cxt = ((self.h[i] + (self.c8 as u32 & mask as u32)) & (cr.c - 1) as u32) as usize;
                    let w = *cr.a16.at(cr.cxt) as i32;
                    self.p[i] = ((w * self.p[self.z.header[cp + 2] as usize]
                        + (65536 - w) * self.p[self.z.header[cp + 3] as usize])
                        >> 16) as i32;
                    self.p[i] = clamp2k(self.p[i]);
                }
                7 => {
                    let m = self.z.header[cp + 3] as usize;
                    let mask = self.z.header[cp + 5] as u32;
                    cr.cxt = (self.h[i] + (self.c8 as u32 & mask)) as usize;
                    cr.cxt = (cr.cxt & (cr.c - 1)) * m;
                    let mut sum = 0i32;
                    for j in 0..m {
                        let wt = (*cr.cm.at(cr.cxt + j) >> 8) as i32;
                        sum += wt * self.p[self.z.header[cp + 2] as usize + j];
                    }
                    self.p[i] = clamp2k(sum >> 8);
                }
                8 => {
                    if self.c8 == 1 || (self.c8 & 0xf0) == 16 {
                        cr.c = find(&mut cr.ht, (self.z.header[cp + 1] + 2) as usize, self.h[i] + 16 * self.c8 as u32);
                    }
                    cr.cxt = *cr.ht.at(cr.c + (self.hmap4 as usize & 15)) as usize;
                    let wt0 = *cr.cm.at(cr.cxt * 2) as i32;
                    let wt1 = *cr.cm.at(cr.cxt * 2 + 1) as i32;
                    self.p[i] = clamp2k(((wt0 * self.p[self.z.header[cp + 2] as usize] + wt1 * 64) >> 16) as i32);
                }
                9 => {
                    let base = cr.cm.size() >> 5;
                    let mask = base - 1;
                    cr.cxt = ((self.h[i].wrapping_add(self.c8 as u32)) as usize & mask) * 32;
                    let mut pq = self.p[self.z.header[cp + 2] as usize] + 992;
                    if pq < 0 {
                        pq = 0;
                    }
                    if pq > 1983 {
                        pq = 1983;
                    }
                    let wt = (pq & 63) as usize;
                    pq >>= 6;
                    cr.cxt += pq as usize;
                    let v0 = *cr.cm.at(cr.cxt) >> 10;
                    let v1 = *cr.cm.at(cr.cxt + 1) >> 10;
                    let p = ((v0 * (64 - wt as u32) + v1 * wt as u32) >> 13) as i32;
                    self.p[i] = stretch(p, &self.stretcht);
                    cr.cxt += wt >> 5;
                }
                _ => {}
            }
            cp += COMPSIZE[typ as usize] as usize;
        }
        squash(self.p[n - 1], &self.squasht)
    }

    fn update0(&mut self, y: i32) {
        let n = self.z.header[6] as usize;
        let mut cp = 7usize;
        for i in 0..n {
            let typ = self.z.header[cp];
            let cr = &mut self.comp[i];
            match typ {
                1 => {}
                2 => train(cr, y, &self.dt),
                3 => {
                    let idx = cr.c + (self.hmap4 as usize & 15);
                    let state = *cr.ht.at(idx);
                    *cr.ht.at_mut(idx) = self.st.next(state, y as u8);
                    let p = *cr.cm.at(cr.cxt);
                    let err = y * 32767 - ((p >> 8) as i32);
                    let newp = p.wrapping_add(((err >> 2) as u32) as u32);
                    *cr.cm.at_mut(cr.cxt) = newp;
                }
                4 => {
                    if cr.c as i32 != y {
                        cr.a = 0;
                    }
                    let v = cr.ht.at_mut(cr.limit);
                    *v = v.wrapping_mul(2).wrapping_add(y as u8);
                    cr.cxt += 1;
                    if cr.cxt == 8 {
                        cr.cxt = 0;
                        cr.limit = (cr.limit + 1) & (cr.ht.size() - 1);
                        if cr.a == 0 {
                            cr.b = cr.limit.wrapping_sub(*cr.cm.at_mod(self.h[i] as usize) as usize);
                            if cr.b & (cr.ht.size() - 1) != 0 {
                                while cr.a < 255
                                    && cr.ht.at_mod(cr.limit - cr.a - 1)
                                        == cr.ht.at_mod(cr.limit - cr.a - cr.b - 1)
                                {
                                    cr.a += 1;
                                }
                            }
                        } else if cr.a < 255 {
                            cr.a += 1;
                        }
                        *cr.cm.at_mod_mut(self.h[i] as usize) = cr.limit as u32;
                    }
                }
                5 => {}
                6 => {
                    let err = (y * 32767 - squash(self.p[i], &self.squasht)) * self.z.header[cp + 4] as i32 >> 5;
                    let w = *cr.a16.at(cr.cxt) as i32;
                    let delta = (err * (self.p[self.z.header[cp + 2] as usize] - self.p[self.z.header[cp + 3] as usize])
                        + (1 << 12))
                        >> 13;
                    let mut w2 = w + delta;
                    w2 = min(65535, max(0, w2));
                    *cr.a16.at_mut(cr.cxt) = w2 as u16;
                }
                7 => {
                    let m = self.z.header[cp + 3] as usize;
                    let err = (y * 32767 - squash(self.p[i], &self.squasht)) * self.z.header[cp + 4] as i32 >> 4;
                    for j in 0..m {
                        let wt = cr.cm.at_mut(cr.cxt + j);
                        let delta = (err * self.p[self.z.header[cp + 2] as usize + j] + (1 << 12)) >> 13;
                        let newv = clamp512k((*wt as i32) + delta) as u32;
                        *wt = newv;
                    }
                }
                8 => {
                    let err = y * 32767 - squash(self.p[i], &self.squasht);
                    let idx0 = cr.cxt * 2;
                    let idx1 = idx0 + 1;
                    let wt0 = *cr.cm.at(idx0) as i32;
                    let wt1 = *cr.cm.at(idx1) as i32;
                    let new0 =
                        clamp512k(wt0 + ((err * self.p[self.z.header[cp + 2] as usize] + (1 << 12)) >> 13)) as u32;
                    let new1 = clamp512k(wt1 + ((err + 16) >> 5)) as u32;
                    *cr.cm.at_mut(idx0) = new0;
                    *cr.cm.at_mut(idx1) = new1;
                    let idx = cr.c + (self.hmap4 as usize & 15);
                    *cr.ht.at_mut(idx) = self.st.next(cr.cxt as u8, y as u8);
                }
                9 => train(cr, y, &self.dt),
                _ => {}
            }
            cp += COMPSIZE[typ as usize] as usize;
        }

        self.c8 = self.c8 + self.c8 + y;
        if self.c8 >= 256 {
            let mut out = Vec::new();
            self.z.run((self.c8 - 256) as u32, &mut out);
            self.hmap4 = 1;
            self.c8 = 1;
            for i in 0..n {
                self.h[i] = self.z.h(i);
            }
        } else if self.c8 >= 16 && self.c8 < 32 {
            self.hmap4 = (self.hmap4 & 0xf) << 5 | (y << 4) | 1;
        } else {
            self.hmap4 = (self.hmap4 & 0x1f0) | (((self.hmap4 & 0xf) * 2 + y) & 0xf);
        }
    }
}

fn squash(x: i32, table: &[u16; 4096]) -> i32 {
    let x = x.clamp(-2048, 2047) + 2048;
    table[x as usize] as i32
}

fn stretch(x: i32, table: &[i16; 32768]) -> i32 {
    let x = x.clamp(0, 32767) as usize;
    table[x] as i32
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

fn train(cr: &mut Component, y: i32, dt: &[i32; 1024]) {
    let pn = cr.cm.at_mut(cr.cxt);
    let count = (*pn & 0x3ff) as usize;
    let error = y * 32767 - ((*pn >> 17) as i32);
    let update = (error * dt[count]) & -1024;
    *pn = pn.wrapping_add(update as u32).wrapping_add((count < cr.limit) as u32);
}

fn find(ht: &mut Array<u8>, sizebits: usize, cxt: u32) -> usize {
    let size = 16usize << sizebits;
    let chk = ((cxt >> sizebits) & 255) as u8;
    let h0 = (cxt as usize * 16) & (size - 16);
    if *ht.at(h0) == chk {
        return h0;
    }
    let h1 = h0 ^ 16;
    if *ht.at(h1) == chk {
        return h1;
    }
    let h2 = h0 ^ 32;
    if *ht.at(h2) == chk {
        return h2;
    }
    let r = if *ht.at(h0 + 1) <= *ht.at(h1 + 1) && *ht.at(h0 + 1) <= *ht.at(h2 + 1) {
        h0
    } else if *ht.at(h1 + 1) < *ht.at(h2 + 1) {
        h1
    } else {
        h2
    };
    for i in 0..16 {
        *ht.at_mut(r + i) = 0;
    }
    *ht.at_mut(r) = chk;
    r
}

struct Encoder {
    low: u32,
    high: u32,
    pr: Predictor,
}

impl Encoder {
    fn new(z: ZPAQL) -> Self {
        Self {
            low: 1,
            high: 0xFFFF_FFFF,
            pr: Predictor::new(z),
        }
    }

    fn init(&mut self) {
        self.low = 1;
        self.high = 0xFFFF_FFFF;
        self.pr.init();
    }

    fn encode(&mut self, out: &mut ByteWriter, y: i32, p: i32) {
        let mid = self.low.wrapping_add((((self.high - self.low) as u64 * p as u64) >> 16) as u32);
        if y != 0 {
            self.high = mid;
        } else {
            self.low = mid.wrapping_add(1);
        }
        while (self.high ^ self.low) < 0x0100_0000 {
            out.put((self.high >> 24) as i32);
            self.high = (self.high << 8) | 255;
            self.low <<= 8;
            if self.low == 0 {
                self.low = 1;
            }
        }
    }

    fn compress(&mut self, out: &mut ByteWriter, c: i32) {
        if self.pr.is_modeled() {
            if c == -1 {
                self.encode(out, 1, 0);
            } else {
                self.encode(out, 0, 0);
                for i in (0..8).rev() {
                    let p = self.pr.predict() * 2 + 1;
                    let y = (c >> i) & 1;
                    self.encode(out, y, p);
                    self.pr.update(y);
                }
            }
        } else {
            error("unmodeled encode not supported");
        }
    }
}

struct Decoder {
    low: u32,
    high: u32,
    curr: u32,
    pr: Predictor,
}

impl Decoder {
    fn new(z: ZPAQL) -> Self {
        Self {
            low: 1,
            high: 0xFFFF_FFFF,
            curr: 0,
            pr: Predictor::new(z),
        }
    }

    fn init(&mut self) {
        self.pr.init();
        self.low = 1;
        self.high = 0xFFFF_FFFF;
        self.curr = 0;
    }

    fn load(&mut self, input: &mut ByteReader) {
        self.curr = 0;
        for _ in 0..4 {
            let c = input.get();
            if c < 0 {
                error("unexpected end of file");
            }
            self.curr = (self.curr << 8) | (c as u32);
        }
    }

    fn decode(&mut self, input: &mut ByteReader, p: i32) -> i32 {
        let mid = self.low.wrapping_add((((self.high - self.low) as u64 * p as u64) >> 16) as u32);
        let y;
        if self.curr <= mid {
            y = 1;
            self.high = mid;
        } else {
            y = 0;
            self.low = mid.wrapping_add(1);
        }
        while (self.high ^ self.low) < 0x0100_0000 {
            self.high = (self.high << 8) | 255;
            self.low <<= 8;
            if self.low == 0 {
                self.low = 1;
            }
            let c = input.get();
            if c < 0 {
                error("unexpected end of file");
            }
            self.curr = (self.curr << 8) | (c as u32);
        }
        y
    }

    fn decompress(&mut self, input: &mut ByteReader) -> i32 {
        if self.pr.is_modeled() {
            if self.decode(input, 0) != 0 {
                if self.curr != 0 {
                    error("decoding end of stream");
                }
                return -1;
            }
            let mut c = 1;
            while c < 256 {
                let p = self.pr.predict() * 2 + 1;
                c = c + c + self.decode(input, p);
                self.pr.update(c & 1);
            }
            return c - 256;
        }
        error("unmodeled decode not supported");
    }
}

// Built-in models extracted from libzpaq (min/mid/max)
const MODEL_MIN: &[u8] = &[
    26, 0, 1, 2, 0, 0, 2, 3, 16, 8, 19, 0, 0, 96, 4, 28, 59, 10, 59, 112, 25, 10, 59, 10, 59, 112, 56, 0,
];

const MODEL_MID: &[u8] = &[
    69, 0, 3, 3, 0, 0, 8, 3, 5, 8, 13, 0, 8, 17, 1, 8, 18, 2, 8, 18, 3, 8, 19, 4, 4, 22, 24, 7, 16, 0, 7, 24,
    255, 0, 17, 104, 74, 4, 95, 1, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25,
    59, 10, 59, 112, 25, 69, 207, 8, 112, 56, 0,
];

const MODEL_MAX: &[u8] = &[
    196, 0, 5, 9, 0, 0, 22, 1, 160, 3, 5, 8, 13, 1, 8, 16, 2, 8, 18, 3, 8, 19, 4, 8, 19, 5, 8, 20, 6, 4, 22, 24,
    3, 17, 8, 19, 9, 3, 13, 3, 13, 3, 13, 3, 14, 7, 16, 0, 15, 24, 255, 7, 8, 0, 16, 10, 255, 6, 0, 15, 16, 24, 0, 9,
    8, 17, 32, 255, 6, 8, 17, 18, 16, 255, 9, 16, 19, 32, 255, 6, 0, 19, 20, 16, 0, 0, 17, 104, 74, 4, 95, 2, 59, 112, 10, 25,
    59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 10, 59, 112, 10, 25, 59, 112, 10, 25, 69, 183, 32, 239, 64, 47,
    14, 231, 91, 47, 10, 25, 60, 26, 48, 134, 151, 20, 112, 63, 9, 70, 223, 0, 39, 3, 25, 112, 26, 52, 25, 25, 74, 10, 4, 59, 112, 25,
    10, 4, 59, 112, 25, 10, 4, 59, 112, 25, 65, 143, 212, 72, 4, 59, 112, 8, 143, 216, 8, 68, 175, 60, 60, 25, 69, 207, 9, 112, 25, 25,
    25, 25, 25, 112, 56, 0,
];
