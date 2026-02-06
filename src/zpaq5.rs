//! ZPAQ core compressor/decompressor (in-memory), ported from libzpaq.
//!
//! This implements the ZPAQ block format and context-mixing arithmetic coder
//! using the built-in max model (level 3) headers from libzpaq.

use crate::zpaq_consts::*;

use std::cmp::{max, min};

pub fn compress_text(data: &[u8]) -> Vec<u8> {
    compress_level5_block(data)
}

pub fn decompress_text(data: &[u8]) -> Vec<u8> {
    decompress_block(data)
}

// Level 5 (method "5") using libzpaq config + compiler.
pub fn compress_text_level5(data: &[u8]) -> Vec<u8> {
    compress_level5_block(data)
}

pub fn decompress_text_level5(data: &[u8]) -> Vec<u8> {
    decompress_block(data)
}

#[derive(Debug)]
struct ZpaqError(&'static str);

fn error(msg: &str) -> ! {
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

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn peek(&self) -> i32 {
        if self.pos >= self.data.len() {
            return -1;
        }
        self.data[self.pos] as i32
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

use std::cell::RefCell;
use std::rc::Rc;

type SharedOutput = Rc<RefCell<Vec<u8>>>;

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
    output: Option<SharedOutput>,
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
            output: None,
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
        self.output = None;
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
            self.flush();
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

    fn flush(&mut self) {
        if self.bufptr == 0 {
            return;
        }
        if let Some(ref out) = self.output {
            out.borrow_mut().extend_from_slice(&self.outbuf[..self.bufptr]);
        }
        self.bufptr = 0;
    }

    fn set_output(&mut self, out: Option<SharedOutput>) {
        self.output = out;
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
        let end = input.get();
        if end != 0 {
            error("missing HCOMP END");
        }
        self.header[self.hend] = 0;
        self.hend += 1;
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

    fn run_hcomp(&mut self, input: u32, out: &mut Vec<u8>) {
        self.pc = self.hbegin;
        self.a = input;
        while self.execute() {}
        self.take_output(out);
    }

    fn run_pcomp(&mut self, input: i32) {
        self.pc = self.hbegin;
        self.a = input as u32;
        while self.execute() {}
    }

    fn exec_error(&self, op: u8, pc: usize, note: &str) -> ! {
        error(&format!(
            "ZPAQL execution error ({}) op={} pc={} hbegin={} hend={}",
            note, op, pc, self.hbegin, self.hend
        ));
    }

    fn execute(&mut self) -> bool {
        let op = self.header[self.pc];
        self.pc += 1;
        match op {
            0 => self.exec_error(op, self.pc - 1, "op=error"),
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
                if self.f != 0 {
                    self.pc = ((self.pc as i32) + ((off + 128) & 255) - 127) as usize;
                } else {
                    self.pc += 1;
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
                if self.f == 0 {
                    self.pc = ((self.pc as i32) + ((off + 128) & 255) - 127) as usize;
                } else {
                    self.pc += 1;
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
            128 => self.a = self.a.wrapping_add(self.a),
            129 => self.a = self.a.wrapping_add(self.b),
            130 => self.a = self.a.wrapping_add(self.c),
            131 => self.a = self.a.wrapping_add(self.d),
            132 => self.a = self.a.wrapping_add(*self.m_at(self.b) as u32),
            133 => self.a = self.a.wrapping_add(*self.m_at(self.c) as u32),
            134 => self.a = self.a.wrapping_add(*self.h_at(self.d)),
            135 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_add(v);
            }
            136 => self.a = self.a.wrapping_sub(self.a),
            137 => self.a = self.a.wrapping_sub(self.b),
            138 => self.a = self.a.wrapping_sub(self.c),
            139 => self.a = self.a.wrapping_sub(self.d),
            140 => self.a = self.a.wrapping_sub(*self.m_at(self.b) as u32),
            141 => self.a = self.a.wrapping_sub(*self.m_at(self.c) as u32),
            142 => self.a = self.a.wrapping_sub(*self.h_at(self.d)),
            143 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_sub(v);
            }
            144 => self.a = self.a.wrapping_mul(self.a),
            145 => self.a = self.a.wrapping_mul(self.b),
            146 => self.a = self.a.wrapping_mul(self.c),
            147 => self.a = self.a.wrapping_mul(self.d),
            148 => self.a = self.a.wrapping_mul(*self.m_at(self.b) as u32),
            149 => self.a = self.a.wrapping_mul(*self.m_at(self.c) as u32),
            150 => self.a = self.a.wrapping_mul(*self.h_at(self.d)),
            151 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = self.a.wrapping_mul(v);
            }
            152 => {
                let v = self.a;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            153 => {
                let v = self.b;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            154 => {
                let v = self.c;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            155 => {
                let v = self.d;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            156 => {
                let v = *self.m_at(self.b) as u32;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            157 => {
                let v = *self.m_at(self.c) as u32;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            158 => {
                let v = *self.h_at(self.d);
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            159 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = if v == 0 { 0 } else { self.a / v };
            }
            160 => {
                let v = self.a;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            161 => {
                let v = self.b;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            162 => {
                let v = self.c;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            163 => {
                let v = self.d;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            164 => {
                let v = *self.m_at(self.b) as u32;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            165 => {
                let v = *self.m_at(self.c) as u32;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            166 => {
                let v = *self.h_at(self.d);
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            167 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a = if v == 0 { 0 } else { self.a % v };
            }
            168 => self.a &= self.a,
            169 => self.a &= self.b,
            170 => self.a &= self.c,
            171 => self.a &= self.d,
            172 => self.a &= *self.m_at(self.b) as u32,
            173 => self.a &= *self.m_at(self.c) as u32,
            174 => self.a &= *self.h_at(self.d),
            175 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a &= v;
            }
            176 => self.a &= !self.a,
            177 => self.a &= !self.b,
            178 => self.a &= !self.c,
            179 => self.a &= !self.d,
            180 => self.a &= !(*self.m_at(self.b) as u32),
            181 => self.a &= !(*self.m_at(self.c) as u32),
            182 => self.a &= !(*self.h_at(self.d)),
            183 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a &= !v;
            }
            184 => self.a |= self.a,
            185 => self.a |= self.b,
            186 => self.a |= self.c,
            187 => self.a |= self.d,
            188 => self.a |= *self.m_at(self.b) as u32,
            189 => self.a |= *self.m_at(self.c) as u32,
            190 => self.a |= *self.h_at(self.d),
            191 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a |= v;
            }
            192 => self.a ^= self.a,
            193 => self.a ^= self.b,
            194 => self.a ^= self.c,
            195 => self.a ^= self.d,
            196 => self.a ^= *self.m_at(self.b) as u32,
            197 => self.a ^= *self.m_at(self.c) as u32,
            198 => self.a ^= *self.h_at(self.d),
            199 => {
                let v = self.header[self.pc] as u32;
                self.pc += 1;
                self.a ^= v;
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
                    self.exec_error(op, self.pc, "lj out of range");
                }
            }
            _ => self.exec_error(op, self.pc - 1, "unknown op"),
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
                            cr.b =
                                cr.limit
                                    .wrapping_sub(*cr.cm.at_mod(self.h[i] as usize) as usize);
                            if cr.b & (cr.ht.size() - 1) != 0 {
                                while cr.a < 255 {
                                    let idx1 = cr.limit.wrapping_sub(cr.a).wrapping_sub(1);
                                    let idx2 = cr
                                        .limit
                                        .wrapping_sub(cr.a)
                                        .wrapping_sub(cr.b)
                                        .wrapping_sub(1);
                                    if cr.ht.at_mod(idx1) != cr.ht.at_mod(idx2) {
                                        break;
                                    }
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
            self.z.run_hcomp((self.c8 - 256) as u32, &mut out);
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
    buf: Vec<u8>,
}

impl Encoder {
    fn new(z: ZPAQL) -> Self {
        Self {
            low: 1,
            high: 0xFFFF_FFFF,
            pr: Predictor::new(z),
            buf: Vec::new(),
        }
    }

    fn init(&mut self) {
        self.low = 1;
        self.high = 0xFFFF_FFFF;
        self.pr.init();
        if !self.pr.is_modeled() {
            self.low = 0;
            self.buf.resize(1 << 16, 0);
        }
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
            if self.low != 0 && (c < 0 || self.low as usize == self.buf.len()) {
                out.put((self.low >> 24) as i32);
                out.put((self.low >> 16) as i32);
                out.put((self.low >> 8) as i32);
                out.put((self.low) as i32);
                out.write(&self.buf[..self.low as usize]);
                self.low = 0;
            }
            if c >= 0 {
                self.buf[self.low as usize] = c as u8;
                self.low += 1;
            }
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
            if self.curr == 0 {
                for _ in 0..4 {
                    let c = input.get();
                    if c < 0 {
                        error("unexpected end of file");
                    }
                    self.curr = (self.curr << 8) | (c as u32);
                }
            }
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
        } else {
            if self.curr == 0 {
                for _ in 0..4 {
                    let c = input.get();
                    if c < 0 {
                        error("unexpected end of file");
                    }
                    self.curr = (self.curr << 8) | (c as u32);
                }
                if self.curr == 0 {
                    return -1;
                }
            }
            self.curr -= 1;
            return input.get();
        }
    }
}

// Built-in models extracted from libzpaq (min/mid/max)
const ZPAQ_TAG: [u8; 13] = [0x37, 0x6b, 0x53, 0x74, 0xa0, 0x31, 0x83, 0xd3, 0x8c, 0xb2, 0x28, 0xb0, 0xd3];

const MODEL_MIN: &[u8] = &[
    26, 0, 1, 2, 0, 0, 2, 3, 16, 8, 19, 0, 0, 96, 4, 28, 59, 10, 59, 112, 25, 10, 59, 10, 59, 112, 56, 0,
];

fn dump_hcomp(z: &ZPAQL, max_ops: usize) {
    eprintln!(
        "HCOMP bytes: {} (hbegin={}, hend={})",
        z.hend - z.hbegin,
        z.hbegin,
        z.hend
    );
    let mut pc = z.hbegin;
    let mut count = 0usize;
    while pc < z.hend && count < max_ops {
        let op = z.header[pc] as usize;
        let name = OPCODELIST.get(op).copied().unwrap_or("?");
        let start = pc;
        pc += 1;
        let mut extra = String::new();
        if op == 255 {
            if pc + 1 < z.hend {
                let lo = z.header[pc];
                let hi = z.header[pc + 1];
                extra = format!(" {} {}", lo, hi);
            }
            pc += 2;
        } else if (op & 7) == 7 {
            if pc < z.hend {
                let v = z.header[pc];
                extra = format!(" {}", v);
            }
            pc += 1;
        }
        eprintln!(
            "{:04} {:3} {:<6}{}",
            start - z.hbegin,
            op,
            name,
            extra
        );
        count += 1;
    }
}

fn compress_level5_block(data: &[u8]) -> Vec<u8> {
    let mut out = ByteWriter::new();

    // Build config for method "5" (default block size from input length).
    let (config, mut args) = make_config_level5(data);
    let mut z = ZPAQL::new();
    let mut pz = ZPAQL::new();
    let mut pcomp_cmd = Vec::new();
    compile_config(&config, &mut args, &mut z, &mut pz, Some(&mut pcomp_cmd));
    if std::env::var("ZPAQ_DUMP").is_ok() {
        dump_hcomp(&z, 200);
    }

    // Optional tag for compatibility with zpaq (13 bytes)
    out.write(&ZPAQ_TAG);

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

    // Write PCOMP program (if any) into the arithmetic stream.
    let mut enc = Encoder::new(z.clone());
    enc.init();
    post_process(&mut enc, &mut out, &pz);

    // Encode data bytes
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

    // End block
    out.put(255);

    out.into_inner()
}

fn decompress_block(data: &[u8]) -> Vec<u8> {
    let mut reader = ByteReader::new(data);

    // Optional 13-byte tag
    if reader.remaining() >= ZPAQ_TAG.len() {
        let mut tag = [0u8; 13];
        for i in 0..13 {
            let c = reader.get();
            if c < 0 {
                error("unexpected end of file");
            }
            tag[i] = c as u8;
        }
        if tag != ZPAQ_TAG {
            // Not a tag; rewind
            reader.pos = 0;
        }
    }

    // Read zPQ header
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
    let _level = tag[3];
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

    // Decode stream with postprocessor
    let mut dec = Decoder::new(z.clone());
    dec.init();

    let out = Rc::new(RefCell::new(Vec::new()));
    let mut pp = PostProcessor::new(out.clone());
    pp.init(z.header[4] as i32, z.header[5] as i32);

    // Load PCOMP (PASS/PROG) first
    while (pp.get_state() & 3) != 1 {
        let c = dec.decompress(&mut reader);
        pp.write(c);
    }

    loop {
        let c = dec.decompress(&mut reader);
        pp.write(c);
        if c < 0 {
            break;
        }
    }
    pp.z.flush();

    // Skip end marker (0 0 0 0 + 254/253)
    let _ = reader.get();
    let _ = reader.get();
    let _ = reader.get();
    let _ = reader.get();
    let _ = reader.get();

    // Optional end block 255
    if reader.peek() == 255 {
        let _ = reader.get();
    }

    drop(pp);
    Rc::try_unwrap(out).unwrap().into_inner()
}

struct PostProcessor {
    state: i32,
    hsize: i32,
    ph: i32,
    pm: i32,
    z: ZPAQL,
    output: SharedOutput,
}

impl PostProcessor {
    fn new(output: SharedOutput) -> Self {
        let mut z = ZPAQL::new();
        z.set_output(Some(output.clone()));
        Self {
            state: 0,
            hsize: 0,
            ph: 0,
            pm: 0,
            z,
            output,
        }
    }

    fn init(&mut self, h: i32, m: i32) {
        self.state = 0;
        self.hsize = 0;
        self.ph = h;
        self.pm = m;
        self.z.clear();
        self.z.set_output(Some(self.output.clone()));
    }

    fn get_state(&self) -> i32 {
        self.state
    }

    fn write(&mut self, c: i32) -> i32 {
        match self.state {
            0 => {
                if c < 0 {
                    error("Unexpected EOS");
                }
                self.state = c + 1; // 1=PASS, 2=PROG
                if self.state > 2 {
                    error("unknown post processing type");
                }
                if self.state == 1 {
                    self.z.clear();
                    self.z.set_output(Some(self.output.clone()));
                }
            }
            1 => {
                self.z.outc(c);
            }
            2 => {
                if c < 0 {
                    error("Unexpected EOS");
                }
                self.hsize = c;
                self.state = 3;
            }
            3 => {
                if c < 0 {
                    error("Unexpected EOS");
                }
                self.hsize += c * 256;
                if self.hsize < 1 {
                    error("Empty PCOMP");
                }
                self.z.header.resize(self.hsize as usize + 300, 0);
                self.z.cend = 8;
                self.z.hbegin = self.z.cend + 128;
                self.z.hend = self.z.hbegin;
                self.z.header[4] = self.ph as u8;
                self.z.header[5] = self.pm as u8;
                self.state = 4;
            }
            4 => {
                if c < 0 {
                    error("Unexpected EOS");
                }
                self.z.header[self.z.hend] = c as u8;
                self.z.hend += 1;
                if (self.z.hend - self.z.hbegin) == self.hsize as usize {
                    let hsize = self.z.cend - 2 + self.z.hend - self.z.hbegin;
                    self.z.header[0] = (hsize & 255) as u8;
                    self.z.header[1] = (hsize >> 8) as u8;
                    self.z.initp();
                    self.state = 5;
                }
            }
            5 => {
                self.z.run_pcomp(c);
                if c < 0 {
                    self.z.flush();
                }
            }
            _ => {}
        }
        self.state
    }
}

fn post_process(enc: &mut Encoder, out: &mut ByteWriter, pz: &ZPAQL) {
    let len = pz.hend as i32 - pz.hbegin as i32;
    if len > 0 {
        enc.compress(out, 1);
        enc.compress(out, (len & 255) as i32);
        enc.compress(out, ((len >> 8) & 255) as i32);
        for i in pz.hbegin..pz.hend {
            enc.compress(out, pz.header[i] as i32);
        }
    } else {
        enc.compress(out, 0);
    }
}

fn make_config_level5(data: &[u8]) -> (String, [i32; 9]) {
    let method = "5".to_string();
    let mut args = [0i32; 9];
    let config = expand_method(&method, data, &mut args);
    (config, args)
}

fn expand_method(method_in: &str, data: &[u8], args: &mut [i32; 9]) -> String {
    let mut method = method_in.to_string();
    let n = data.len() as u32;
    let arg0 = std::cmp::max(lg(n + 4095) - 20, 0);

    // Parse type from digit method "LB,R,t"
    let mut typ: i32 = 0;
    if method.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        let mut commas = 0usize;
        let mut arg = [0i32; 4];
        for ch in method.chars().skip(1) {
            if ch == ',' || ch == '.' {
                commas += 1;
                if commas >= 4 {
                    break;
                }
            } else if ch.is_ascii_digit() {
                arg[commas] = arg[commas] * 10 + (ch as i32 - '0' as i32);
            }
        }
        if commas == 0 {
            typ = 512;
        } else {
            typ = arg[1] * 4 + arg[2];
        }
    }

    if method.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        let level = method.as_bytes()[0] - b'0';
        let doe8 = (typ & 2) * 2;
        method = format!("x{}", arg0);
        let htsz = format!(",{}", 19 + arg0 + if arg0 <= 6 { 1 } else { 0 });
        let sasz = format!(",{}", 21 + arg0);

        if level == 0 {
            method = format!("0{},0", arg0);
        } else if level == 1 {
            if typ < 40 {
                method.push_str(",0");
            } else {
                method.push_str(&format!(",{},", 1 + doe8));
                if typ < 80 {
                    method.push_str("4,0,1,15");
                } else if typ < 128 {
                    method.push_str("4,0,2,16");
                } else if typ < 256 {
                    method.push_str(&format!("4,0,2{}", htsz));
                } else if typ < 960 {
                    method.push_str(&format!("5,0,3{}", htsz));
                } else {
                    method.push_str(&format!("6,0,3{}", htsz));
                }
            }
        } else if level == 2 {
            if typ < 32 {
                method.push_str(",0");
            } else {
                method.push_str(&format!(",{},", 1 + doe8));
                if typ < 64 {
                    method.push_str(&format!("4,0,3{}", htsz));
                } else {
                    method.push_str(&format!("4,0,7{},1", sasz));
                }
            }
        } else if level == 3 {
            if typ < 20 {
                method.push_str(",0");
            } else if typ < 48 {
                method.push_str(&format!(",{},4,0,3{}", 1 + doe8, htsz));
            } else if typ >= 640 || (typ & 1) != 0 {
                method.push_str(&format!(",{}ci1", 3 + doe8));
            } else {
                method.push_str(&format!(",{},12,0,7{},1c0,0,511i2", 2 + doe8, sasz));
            }
        } else if level == 4 {
            if typ < 12 {
                method.push_str(",0");
            } else if typ < 24 {
                method.push_str(&format!(",{},4,0,3{}", 1 + doe8, htsz));
            } else if typ < 48 {
                method.push_str(&format!(",{},5,0,7{}1c0,0,511", 2 + doe8, sasz));
            } else if typ < 900 {
                method.push_str(&format!(",{}ci1,1,1,1,2a", doe8));
                if (typ & 1) != 0 {
                    method.push('w');
                }
                method.push('m');
            } else {
                method.push_str(&format!(",{}ci1", 3 + doe8));
            }
        } else {
            // level 5..9: slow CM with lots of models
            method.push_str(&format!(",{}", doe8));
            if (typ & 1) != 0 {
                method.push_str("w2c0,1010,255i1");
            } else {
                method.push_str("w1i1");
            }
            method.push_str("c256ci1,1,1,1,1,1,2a");

            const NR: usize = 1 << 12;
            let mut pt = [0i32; 256];
            let mut r = vec![0i32; NR];
            for (i, &b) in data.iter().enumerate() {
                let k = i as i32 - pt[b as usize];
                if k > 0 && (k as usize) < NR {
                    r[k as usize] += 1;
                }
                pt[b as usize] = i as i32;
            }

            let mut n1 = data.len() as i32 - r[1] - r[2] - r[3];
            for _ in 0..2 {
                let mut period = 0usize;
                let mut score = 0.0f64;
                let mut t = 0i32;
                for j in 5..NR {
                    if t >= n1 {
                        break;
                    }
                    let s = r[j] as f64 / (256.0 + (n1 - t) as f64);
                    if s > score {
                        score = s;
                        period = j;
                    }
                    t += r[j];
                }
                if period > 4 && score > 0.1 {
                    method.push_str(&format!("c0,0,{},255i1", 999 + period));
                    if period <= 255 {
                        method.push_str(&format!("c0,{}i1", period));
                    }
                    n1 -= r[period];
                    r[period] = 0;
                } else {
                    break;
                }
            }
            method.push_str("c0,2,0,255i1c0,3,0,0,255i1c0,4,0,0,0,255i1mm16ts19t0");
        }
    }

    make_config(&method, args)
}

fn make_config(method: &str, args: &mut [i32; 9]) -> String {
    let bytes = method.as_bytes();
    if bytes.is_empty() {
        error("empty method");
    }
    let typ = bytes[0] as char;
    if typ != 'x' && typ != 's' && typ != 'i' && typ != '0' {
        error("method must begin with 0..5, x, s");
    }

    // Parse "{x|s|i|0}N1,N2...N9" into args[0..8]
    for v in args.iter_mut() {
        *v = 0;
    }
    let mut pos = 1usize;
    let mut i = 0usize;
    while i < 9 && pos < bytes.len() {
        let c = bytes[pos] as char;
        if c.is_ascii_digit() || c == ',' || c == '.' {
            if c.is_ascii_digit() {
                args[i] = args[i] * 10 + (c as i32 - '0' as i32);
            } else {
                i += 1;
                if i < 9 {
                    args[i] = 0;
                }
            }
            pos += 1;
        } else {
            break;
        }
    }
    let rest = &method[pos..];

    if typ == '0' {
        return "comp 0 0 0 0 0 hcomp end\n".to_string();
    }

    let level = args[1] & 3;
    let doe8 = args[1] >= 4 && args[1] <= 7;

    if level != 0 || doe8 {
        error("Only level-0 (no preprocessing) is supported in zpaq5 minimal path");
    }

    let mut hdr = "comp 9 16 0 0 ".to_string();
    let pcomp = "end\n".to_string();

    // Build context model (comp, hcomp)
    let mut ncomp = 0i32;
    let membits = args[0] + 20;
    let mut sb = 5i32;
    let mut comp = String::new();
    let mut hcomp = String::from("hcomp\nc-- *c=a a+= 255 d=a *d=c\n");

    let mut idx = 0usize;
    let rest_bytes = rest.as_bytes();
    while idx < rest_bytes.len() && ncomp < 254 {
        let cmd = rest_bytes[idx] as char;
        idx += 1;
        let mut v: Vec<i32> = vec![cmd as i32];
        if idx < rest_bytes.len() && (rest_bytes[idx] as char).is_ascii_digit() {
            v.push(0);
            while idx < rest_bytes.len() {
                let c = rest_bytes[idx] as char;
                if c.is_ascii_digit() {
                    let last = v.len() - 1;
                    v[last] = v[last] * 10 + (c as i32 - '0' as i32);
                    idx += 1;
                } else if c == ',' || c == '.' {
                    v.push(0);
                    idx += 1;
                } else {
                    break;
                }
            }
        }

        match cmd {
            'c' => {
                while v.len() < 3 {
                    v.push(0);
                }
                comp.push_str(&format!("{} ", ncomp));
                sb = 11;
                if v[2] < 256 {
                    sb += lg(v[2] as u32);
                } else {
                    sb += 6;
                }
                for &x in v.iter().skip(3) {
                    if x < 512 {
                        sb += (nbits(x as u32) * 3 / 4) as i32;
                    }
                }
                if sb > membits {
                    sb = membits;
                }
                if v[1] % 1000 == 0 {
                    comp.push_str(&format!("icm {}\n", sb - 6 - v[1] / 1000));
                } else {
                    comp.push_str(&format!(
                        "cm {} {}\n",
                        sb - 2 - v[1] / 1000,
                        v[1] % 1000 - 1
                    ));
                }

                hcomp.push_str(&format!("d= {} *d=0\n", ncomp));
                if v[2] > 1 && v[2] <= 255 {
                    if lg(v[2] as u32) != lg((v[2] - 1) as u32) {
                        hcomp.push_str(&format!("a=c a&= {} hashd\n", v[2] - 1));
                    } else {
                        hcomp.push_str(&format!("a=c a%= {} hashd\n", v[2]));
                    }
                } else if v[2] >= 1000 && v[2] <= 1255 {
                    hcomp.push_str(&format!(
                        "a= 255 a+= {} d=a a=*d a-=c a> 255 if a= 255 endif d= {} hashd\n",
                        v[2] - 1000,
                        ncomp
                    ));
                }

                for (i, &x) in v.iter().skip(3).enumerate() {
                    if i == 0 {
                        hcomp.push_str("b=c ");
                    }
                    if x == 255 {
                        hcomp.push_str("a=*b hashd\n");
                    } else if x > 0 && x < 255 {
                        hcomp.push_str(&format!("a=*b a&= {} hashd\n", x));
                    } else if x >= 256 && x < 512 {
                        hcomp.push_str("a=r 1 a> 1 if\n");
                        hcomp.push_str("  a=r 2 a< 64 if\n");
                        hcomp.push_str("    a=*b ");
                        if x < 511 {
                            hcomp.push_str(&format!("a&= {} ", x - 256));
                        }
                        hcomp.push_str("hashd\n");
                        hcomp.push_str("  else\n");
                        hcomp.push_str("    a>>= 6 hashd a=r 1 hashd\n");
                        hcomp.push_str("  endif\n");
                        hcomp.push_str("else\n");
                        hcomp.push_str("  a= 255 hashd a=r 2 hashd\n");
                        hcomp.push_str("endif\n");
                    } else if x >= 1256 {
                        let v0 = x - 1000;
                        let hi = (v0 >> 8) & 255;
                        let lo = v0 & 255;
                        hcomp.push_str(&format!("a= {} a<<= 8 a+= {} a+=b b=a\n", hi, lo));
                    } else if x > 1000 {
                        hcomp.push_str(&format!("a= {} a+=b b=a\n", x - 1000));
                    }
                    if x < 512 && (i + 4) < v.len() {
                        hcomp.push_str("b++ ");
                    }
                }
                ncomp += 1;
            }
            'm' | 't' | 's' => {
                if ncomp > if cmd == 't' { 1 } else { 0 } {
                    if v.len() <= 1 {
                        v.push(8);
                    }
                    if v.len() <= 2 {
                        v.push(24 + if cmd == 's' { 8 } else { 0 });
                    }
                    if cmd == 's' && v.len() <= 3 {
                        v.push(255);
                    }
                    comp.push_str(&format!("{}", ncomp));
                    sb = 5 + (v[1] * 3 / 4);
                    if cmd == 'm' {
                        comp.push_str(&format!(
                            " mix {} 0 {} {} 255\n",
                            v[1], ncomp, v[2]
                        ));
                    } else if cmd == 't' {
                        comp.push_str(&format!(
                            " mix2 {} {} {} {} 255\n",
                            v[1],
                            ncomp - 1,
                            ncomp - 2,
                            v[2]
                        ));
                    } else {
                        comp.push_str(&format!(
                            " sse {} {} {} {}\n",
                            v[1],
                            ncomp - 1,
                            v[2],
                            v[3]
                        ));
                    }
                    if v[1] > 8 {
                        hcomp.push_str(&format!("d= {} *d=0 b=c a=0\n", ncomp));
                        let mut bits = v[1];
                        while bits >= 16 {
                            hcomp.push_str("a<<= 8 a+=*b");
                            if bits > 16 {
                                hcomp.push_str(" b++");
                            }
                            hcomp.push('\n');
                            bits -= 8;
                        }
                        if bits > 8 {
                            hcomp.push_str(&format!("a<<= 8 a+=*b a>>= {}\n", 16 - bits));
                        }
                        hcomp.push_str("a<<= 8 *d=a\n");
                    }
                    ncomp += 1;
                }
            }
            'i' => {
                if ncomp > 0 {
                    hcomp.push_str(&format!("d= {} b=c a=*d d++\n", ncomp - 1));
                    for idx_i in 1..v.len() {
                        if ncomp >= 254 {
                            break;
                        }
                        for j in 0..(v[idx_i] % 10) {
                            hcomp.push_str("hash ");
                            if idx_i < v.len() - 1 || j < (v[idx_i] % 10) - 1 {
                                hcomp.push_str("b++ ");
                            }
                            sb += 6;
                        }
                        hcomp.push_str("*d=a");
                        if idx_i < v.len() - 1 {
                            hcomp.push_str(" d++");
                        }
                        hcomp.push('\n');
                        if sb > membits {
                            sb = membits;
                        }
                        comp.push_str(&format!(
                            "{} isse {} {}\n",
                            ncomp,
                            sb - 6 - v[idx_i] / 10,
                            ncomp - 1
                        ));
                        ncomp += 1;
                    }
                }
            }
            'a' => {
                if v.len() <= 1 {
                    v.push(24);
                }
                while v.len() < 4 {
                    v.push(0);
                }
                comp.push_str(&format!(
                    "{} match {} {}\n",
                    ncomp,
                    membits - v[3] - 2,
                    membits - v[2]
                ));
                hcomp.push_str(&format!(
                    "d= {} a=*d a*= {} a+=*c a++ *d=a\n",
                    ncomp, v[1]
                ));
                sb = 5 + ((membits - v[2]) * 3 / 4);
                ncomp += 1;
            }
            'w' => {
                if v.len() <= 1 {
                    v.push(1);
                }
                if v.len() <= 2 {
                    v.push(65);
                }
                if v.len() <= 3 {
                    v.push(26);
                }
                if v.len() <= 4 {
                    v.push(223);
                }
                if v.len() <= 5 {
                    v.push(20);
                }
                if v.len() <= 6 {
                    v.push(0);
                }
                comp.push_str(&format!("{} icm {}\n", ncomp, membits - 6 - v[6]));
                for i2 in 1..v[1] {
                    comp.push_str(&format!(
                        "{} isse {} {}\n",
                        ncomp + i2,
                        membits - 6 - v[6],
                        ncomp + i2 - 1
                    ));
                }
                hcomp.push_str(&format!(
                    "a=*c a&= {} a-= {} a&= 255 a< {} if\n",
                    v[4], v[2], v[3]
                ));
                for i2 in 0..v[1] {
                    if i2 == 0 {
                        hcomp.push_str(&format!("  d= {}", ncomp));
                    } else {
                        hcomp.push_str("  d++");
                    }
                    hcomp.push_str(&format!(
                        " a=*d a*= {} a+=*c a++ *d=a\n",
                        v[5]
                    ));
                }
                hcomp.push_str("else\n");
                for i2 in (1..v[1]).rev() {
                    hcomp.push_str(&format!(
                        "  d= {} a=*d d++ *d=a\n",
                        ncomp + i2 - 1
                    ));
                }
                hcomp.push_str(&format!("  d= {} *d=0\nendif\n", ncomp));
                ncomp += v[1] - 1;
                sb = membits - v[6];
                ncomp += 1;
            }
            _ => {}
        }
    }

    format!("{}{}\n{}{}halt\n{}", hdr, ncomp, comp, hcomp, pcomp)
}

fn compile_config(
    input: &str,
    args: &mut [i32; 9],
    hz: &mut ZPAQL,
    pz: &mut ZPAQL,
    mut pcomp_cmd: Option<&mut Vec<u8>>,
) {
    let mut compiler = Compiler::new(input, args, hz, pz, pcomp_cmd.as_deref_mut());
    compiler.compile();
}

fn lg(x: u32) -> i32 {
    if x == 0 {
        return 0;
    }
    (32 - x.leading_zeros()) as i32
}

fn nbits(x: u32) -> i32 {
    x.count_ones() as i32
}

struct Compiler<'a> {
    input: &'a [u8],
    pos: usize,
    line: i32,
    args: &'a [i32; 9],
    hz: *mut ZPAQL,
    pz: *mut ZPAQL,
    out2: Option<&'a mut Vec<u8>>,
    if_stack: Vec<usize>,
    do_stack: Vec<usize>,
    ids: OpcodeIds,
}

struct OpcodeIds {
    post: usize,
    pcomp: usize,
    end: usize,
    if_: usize,
    ifnot: usize,
    else_: usize,
    endif: usize,
    do_: usize,
    while_: usize,
    until: usize,
    forever: usize,
    ifl: usize,
    ifnotl: usize,
    elsel: usize,
    jt: usize,
    jf: usize,
    jmp: usize,
    lj: usize,
}

impl OpcodeIds {
    fn new() -> Self {
        Self {
            post: opcode_index("post"),
            pcomp: opcode_index("pcomp"),
            end: opcode_index("end"),
            if_: opcode_index("if"),
            ifnot: opcode_index("ifnot"),
            else_: opcode_index("else"),
            endif: opcode_index("endif"),
            do_: opcode_index("do"),
            while_: opcode_index("while"),
            until: opcode_index("until"),
            forever: opcode_index("forever"),
            ifl: opcode_index("ifl"),
            ifnotl: opcode_index("ifnotl"),
            elsel: opcode_index("elsel"),
            jt: opcode_index("jt"),
            jf: opcode_index("jf"),
            jmp: opcode_index("jmp"),
            lj: opcode_index("lj"),
        }
    }
}

impl<'a> Compiler<'a> {
    fn new(
        input: &'a str,
        args: &'a [i32; 9],
        hz: &'a mut ZPAQL,
        pz: &'a mut ZPAQL,
        out2: Option<&'a mut Vec<u8>>,
    ) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
            line: 1,
            args,
            hz: hz as *mut ZPAQL,
            pz: pz as *mut ZPAQL,
            out2,
            if_stack: Vec::with_capacity(1024),
            do_stack: Vec::with_capacity(1024),
            ids: OpcodeIds::new(),
        }
    }

    fn compile(&mut self) {
        let hz = unsafe { &mut *self.hz };
        let pz = unsafe { &mut *self.pz };

        hz.clear();
        pz.clear();
        hz.header.resize(68000, 0);

        self.rtoken_str("comp");
        hz.header[2] = self.rtoken_num(0, 255) as u8;
        hz.header[3] = self.rtoken_num(0, 255) as u8;
        hz.header[4] = self.rtoken_num(0, 255) as u8;
        hz.header[5] = self.rtoken_num(0, 255) as u8;
        let n = self.rtoken_num(0, 255) as usize;
        hz.header[6] = n as u8;
        hz.cend = 7;

        for i in 0..n {
            let idx = self.rtoken_num(i as i32, i as i32);
            if idx != i as i32 {
                self.syntax_error("unexpected", Some("component index"));
            }
            let typ = self.rtoken_compname() as u8;
            hz.header[hz.cend] = typ;
            hz.cend += 1;
            let clen = COMPSIZE[typ as usize];
            if clen < 1 || clen > 10 {
                self.syntax_error("invalid component", None);
            }
            for _ in 1..clen {
                let v = self.rtoken_num(0, 255);
                hz.header[hz.cend] = v as u8;
                hz.cend += 1;
            }
        }
        hz.header[hz.cend] = 0;
        hz.cend += 1;
        hz.hbegin = hz.cend + 128;
        hz.hend = hz.hbegin;

        self.rtoken_str("hcomp");
        let op = self.compile_comp_ptr(self.hz);

        let hsize = hz.cend - 2 + hz.hend - hz.hbegin;
        hz.header[0] = (hsize & 255) as u8;
        hz.header[1] = (hsize >> 8) as u8;

        if op == self.ids.post {
            let v = self.rtoken_num(0, 0);
            if v != 0 {
                self.syntax_error("expected 0", None);
            }
            self.rtoken_str("end");
        } else if op == self.ids.pcomp {
            pz.header.resize(68000, 0);
            pz.header[4] = hz.header[4];
            pz.header[5] = hz.header[5];
            pz.cend = 8;
            pz.hbegin = pz.cend + 128;
            pz.hend = pz.hbegin;

            // get pcomp_cmd ending with ';' (case sensitive)
            self.next();
            while self.pos < self.input.len() && self.input[self.pos] as char != ';' {
                if let Some(ref mut out) = self.out2 {
                    out.push(self.input[self.pos]);
                }
                self.pos += 1;
            }
            if self.pos < self.input.len() && self.input[self.pos] as char == ';' {
                self.pos += 1;
            }

            let op2 = self.compile_comp_ptr(self.pz);
            let len = pz.cend - 2 + pz.hend - pz.hbegin;
            pz.header[0] = (len & 255) as u8;
            pz.header[1] = (len >> 8) as u8;
            if op2 != self.ids.end {
                self.syntax_error("expected END", None);
            }
        } else if op != self.ids.end {
            self.syntax_error("expected END or POST 0 END or PCOMP cmd ; ... END", None);
        }
    }

    fn compile_comp_ptr(&mut self, z: *mut ZPAQL) -> usize {
        let z = unsafe { &mut *z };
        let comp_begin = z.hend;
        let mut op = 0usize;
        loop {
            op = self.rtoken_opcode();
            if op == self.ids.post || op == self.ids.pcomp || op == self.ids.end {
                break;
            }

            let mut operand: Option<i32> = None;
            let mut operand2: Option<i32> = None;

            if op == self.ids.if_ {
                op = self.ids.jf;
                operand = Some(0);
                self.if_stack.push(z.hend + 1);
            } else if op == self.ids.ifnot {
                op = self.ids.jt;
                operand = Some(0);
                self.if_stack.push(z.hend + 1);
            } else if op == self.ids.ifl || op == self.ids.ifnotl {
                if op == self.ids.ifl {
                    z.header[z.hend] = self.ids.jt as u8;
                    z.hend += 1;
                } else {
                    z.header[z.hend] = self.ids.jf as u8;
                    z.hend += 1;
                }
                z.header[z.hend] = 3;
                z.hend += 1;
                op = self.ids.lj;
                operand = Some(0);
                operand2 = Some(0);
                self.if_stack.push(z.hend + 1);
            } else if op == self.ids.else_ || op == self.ids.elsel {
                if op == self.ids.else_ {
                    op = self.ids.jmp;
                    operand = Some(0);
                } else {
                    op = self.ids.lj;
                    operand = Some(0);
                    operand2 = Some(0);
                }
                let a = self.if_stack.pop().unwrap_or_else(|| self.syntax_error("IF stack underflow", None));
                if z.header[a - 1] != self.ids.lj as u8 {
                    let mut j = (z.hend as i32 - a as i32 + 1 + if op == self.ids.lj { 1 } else { 0 }) as i32;
                    if j > 127 {
                        self.syntax_error("IF too big, try IFL, IFNOTL", None);
                    }
                    z.header[a] = j as u8;
                } else {
                    let j = z.hend as i32 - comp_begin as i32 + 2 + if op == self.ids.lj { 1 } else { 0 };
                    z.header[a] = (j & 255) as u8;
                    z.header[a + 1] = ((j >> 8) & 255) as u8;
                }
                self.if_stack.push(z.hend + 1);
            } else if op == self.ids.endif {
                let a = self.if_stack.pop().unwrap_or_else(|| self.syntax_error("IF stack underflow", None));
                if z.header[a - 1] != self.ids.lj as u8 {
                    let j = z.hend as i32 - a as i32 - 1;
                    if j > 127 {
                        self.syntax_error("IF too big, try IFL, IFNOTL, ELSEL", None);
                    }
                    z.header[a] = j as u8;
                } else {
                    let j = z.hend as i32 - comp_begin as i32;
                    z.header[a] = (j & 255) as u8;
                    z.header[a + 1] = ((j >> 8) & 255) as u8;
                }
            } else if op == self.ids.do_ {
                self.do_stack.push(z.hend);
            } else if op == self.ids.while_ || op == self.ids.until || op == self.ids.forever {
                let a = self.do_stack.pop().unwrap_or_else(|| self.syntax_error("DO stack underflow", None));
                let mut j = a as i32 - z.hend as i32 - 2;
                if j >= -127 {
                    if op == self.ids.while_ {
                        op = self.ids.jt;
                    } else if op == self.ids.until {
                        op = self.ids.jf;
                    } else {
                        op = self.ids.jmp;
                    }
                    operand = Some((j & 255) as i32);
                } else {
                    j = a as i32 - comp_begin as i32;
                    if op == self.ids.while_ {
                        z.header[z.hend] = self.ids.jf as u8;
                        z.hend += 1;
                        z.header[z.hend] = 3;
                        z.hend += 1;
                    } else if op == self.ids.until {
                        z.header[z.hend] = self.ids.jt as u8;
                        z.hend += 1;
                        z.header[z.hend] = 3;
                        z.hend += 1;
                    }
                    op = self.ids.lj;
                    operand = Some((j & 255) as i32);
                    operand2 = Some((j >> 8) as i32);
                }
            } else if (op & 7) == 7 {
                if op == self.ids.lj {
                    let v = self.rtoken_num(0, 65535);
                    operand = Some((v & 255) as i32);
                    operand2 = Some(((v >> 8) & 255) as i32);
                } else if op == self.ids.jt || op == self.ids.jf || op == self.ids.jmp {
                    let v = self.rtoken_num(-128, 127);
                    operand = Some((v & 255) as i32);
                } else {
                    let v = self.rtoken_num(0, 255);
                    operand = Some(v);
                }
            }

            if op <= 255 {
                z.header[z.hend] = op as u8;
                z.hend += 1;
                if let Some(v) = operand {
                    z.header[z.hend] = v as u8;
                    z.hend += 1;
                }
                if let Some(v) = operand2 {
                    z.header[z.hend] = v as u8;
                    z.hend += 1;
                }
            }

            if z.hend >= z.header.len() - 130 || z.hend - z.hbegin + z.cend - 2 > 65535 {
                self.syntax_error("program too big", None);
            }
        }
        z.header[z.hend] = 0;
        z.hend += 1;
        op
    }

    fn next(&mut self) {
        let mut depth = 0i32;
        loop {
            if self.pos >= self.input.len() {
                self.syntax_error("unexpected end of config", None);
            }
            let c = self.input[self.pos];
            if c == b'\n' {
                self.line += 1;
            }
            if depth > 0 {
                if c == b'(' {
                    depth += 1;
                } else if c == b')' {
                    depth -= 1;
                }
                self.pos += 1;
                continue;
            }
            if c == b'(' {
                depth = 1;
                self.pos += 1;
                continue;
            }
            if c <= b' ' {
                self.pos += 1;
                continue;
            }
            break;
        }
    }

    fn next_token(&mut self) -> Vec<u8> {
        self.next();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos];
            if c <= b' ' || c == b'(' {
                break;
            }
            self.pos += 1;
        }
        self.input[start..self.pos].to_vec()
    }

    fn rtoken_opcode(&mut self) -> usize {
        let tok = self.next_token();
        for (i, op) in OPCODELIST.iter().enumerate() {
            if !op.is_empty() && eq_ignore_ascii_case_bytes(&tok, op.as_bytes()) {
                return i;
            }
        }
        self.syntax_error("unexpected", None);
    }

    fn rtoken_compname(&mut self) -> usize {
        let tok = self.next_token();
        for (i, name) in COMPNAME.iter().enumerate() {
            if !name.is_empty() && eq_ignore_ascii_case_bytes(&tok, name.as_bytes()) {
                return i;
            }
        }
        self.syntax_error("unexpected", None);
    }

    fn rtoken_str(&mut self, s: &str) {
        let tok = self.next_token();
        if !eq_ignore_ascii_case_bytes(&tok, s.as_bytes()) {
            self.syntax_error("expected", Some(s));
        }
    }

    fn rtoken_num(&mut self, low: i32, high: i32) -> i32 {
        let tok = self.next_token();
        let mut s = std::str::from_utf8(&tok).unwrap_or("");
        let mut r: i32 = 0;
        if let Some(rest) = s.strip_prefix('$') {
            if rest.len() >= 1 {
                let n = rest.as_bytes()[0];
                if n >= b'1' && n <= b'9' {
                    let idx = (n - b'1') as usize;
                    if idx < self.args.len() {
                        r += self.args[idx];
                    }
                    if rest.len() > 2 && rest.as_bytes()[1] == b'+' {
                        let tail = &rest[2..];
                        r += tail.parse::<i32>().unwrap_or(0);
                    }
                }
            }
        } else {
            r = s.parse::<i32>().unwrap_or_else(|_| self.syntax_error("expected a number", None));
        }
        if r < low {
            self.syntax_error("number too low", None);
        }
        if r > high {
            self.syntax_error("number too high", None);
        }
        r
    }

    fn syntax_error(&self, msg: &str, expected: Option<&str>) -> ! {
        let mut s = format!("Config line {}: {}", self.line, msg);
        if let Some(exp) = expected {
            s.push_str(&format!(", expected: {}", exp));
        }
        error(&s);
    }
}

fn opcode_index(name: &str) -> usize {
    for (i, op) in OPCODELIST.iter().enumerate() {
        if *op == name {
            return i;
        }
    }
    panic!("opcode not found: {}", name);
}

fn eq_ignore_ascii_case_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_ascii_lowercase() != y.to_ascii_lowercase() {
            return false;
        }
    }
    true
}




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
