//! We achieve 5,188,464 bytes by using pcodec & lpaq1.
//!
//! I translated Mahoney's lpaq1 compressor into rust, since I couldn't find any
//! rust bindings of a good context-mixing compressor. In the contest rules,
//! only one rust file is allowed, no binaries or things like that so I couldn't
//! just drop in the zpaq binary.
//!
//! We use pcodec for numeric columns with clear sequential structure and lpaq1
//! for everything else. We preprocess the repo names with a function Gemini 3
//! Pro suggested, which surprisingly shaves of ~100kb on top of lpaq1.
//!
//! I think my solution can be improved (sub 5 mb even?) but I can't be bothered
//! to look at the problem any further right now. Just using a better compressor
//! (gmix) shaves off 127 kb, but the compression times become 40x higher.
//!
//! Takes 34 seconds to run on my laptop for compression/decompression cycle.

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use pco::standalone::{simple_compress, simple_decompress};
use pco::ChunkConfig;
use pco::PagingSpec::EqualPagesUpTo;
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Write};

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};
use std::error::Error;

/*
lpaq1.cpp file compressor, July 24, 2007.
(C) 2007, Matt Mahoney, matmahoney@yahoo.com

    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

Translated Mahoney's cpp code to rust. A little difficult, because of the
ungodly pointer arithmetic in predictor & hashtable. - kjcao
*/
mod lpaq1 {
    use std::io::{Cursor, Read, Result as IoResult};

    const fn nex(state: usize, sel: usize) -> u8 {
        #[rustfmt::skip]
        const STATE_TABLE: [[u8; 2]; 256] = [
        [  1,  2],[  3,  5],[  4,  6],[  7, 10],[  8, 12],[  9, 13],[ 11, 14], // 0
        [ 15, 19],[ 16, 23],[ 17, 24],[ 18, 25],[ 20, 27],[ 21, 28],[ 22, 29], // 7
        [ 26, 30],[ 31, 33],[ 32, 35],[ 32, 35],[ 32, 35],[ 32, 35],[ 34, 37], // 14
        [ 34, 37],[ 34, 37],[ 34, 37],[ 34, 37],[ 34, 37],[ 36, 39],[ 36, 39], // 21
        [ 36, 39],[ 36, 39],[ 38, 40],[ 41, 43],[ 42, 45],[ 42, 45],[ 44, 47], // 28
        [ 44, 47],[ 46, 49],[ 46, 49],[ 48, 51],[ 48, 51],[ 50, 52],[ 53, 43], // 35
        [ 54, 57],[ 54, 57],[ 56, 59],[ 56, 59],[ 58, 61],[ 58, 61],[ 60, 63], // 42
        [ 60, 63],[ 62, 65],[ 62, 65],[ 50, 66],[ 67, 55],[ 68, 57],[ 68, 57], // 49
        [ 70, 73],[ 70, 73],[ 72, 75],[ 72, 75],[ 74, 77],[ 74, 77],[ 76, 79], // 56
        [ 76, 79],[ 62, 81],[ 62, 81],[ 64, 82],[ 83, 69],[ 84, 71],[ 84, 71], // 63
        [ 86, 73],[ 86, 73],[ 44, 59],[ 44, 59],[ 58, 61],[ 58, 61],[ 60, 49], // 70
        [ 60, 49],[ 76, 89],[ 76, 89],[ 78, 91],[ 78, 91],[ 80, 92],[ 93, 69], // 77
        [ 94, 87],[ 94, 87],[ 96, 45],[ 96, 45],[ 48, 99],[ 48, 99],[ 88,101], // 84
        [ 88,101],[ 80,102],[103, 69],[104, 87],[104, 87],[106, 57],[106, 57], // 91
        [ 62,109],[ 62,109],[ 88,111],[ 88,111],[ 80,112],[113, 85],[114, 87], // 98
        [114, 87],[116, 57],[116, 57],[ 62,119],[ 62,119],[ 88,121],[ 88,121], // 105
        [ 90,122],[123, 85],[124, 97],[124, 97],[126, 57],[126, 57],[ 62,129], // 112
        [ 62,129],[ 98,131],[ 98,131],[ 90,132],[133, 85],[134, 97],[134, 97], // 119
        [136, 57],[136, 57],[ 62,139],[ 62,139],[ 98,141],[ 98,141],[ 90,142], // 126
        [143, 95],[144, 97],[144, 97],[ 68, 57],[ 68, 57],[ 62, 81],[ 62, 81], // 133
        [ 98,147],[ 98,147],[100,148],[149, 95],[150,107],[150,107],[108,151], // 140
        [108,151],[100,152],[153, 95],[154,107],[108,155],[100,156],[157, 95], // 147
        [158,107],[108,159],[100,160],[161,105],[162,107],[108,163],[110,164], // 154
        [165,105],[166,117],[118,167],[110,168],[169,105],[170,117],[118,171], // 161
        [110,172],[173,105],[174,117],[118,175],[110,176],[177,105],[178,117], // 168
        [118,179],[110,180],[181,115],[182,117],[118,183],[120,184],[185,115], // 175
        [186,127],[128,187],[120,188],[189,115],[190,127],[128,191],[120,192], // 182
        [193,115],[194,127],[128,195],[120,196],[197,115],[198,127],[128,199], // 189
        [120,200],[201,115],[202,127],[128,203],[120,204],[205,115],[206,127], // 196
        [128,207],[120,208],[209,125],[210,127],[128,211],[130,212],[213,125], // 203
        [214,137],[138,215],[130,216],[217,125],[218,137],[138,219],[130,220], // 210
        [221,125],[222,137],[138,223],[130,224],[225,125],[226,137],[138,227], // 217
        [130,228],[229,125],[230,137],[138,231],[130,232],[233,125],[234,137], // 224
        [138,235],[130,236],[237,125],[238,137],[138,239],[130,240],[241,125], // 231
        [242,137],[138,243],[130,244],[245,135],[246,137],[138,247],[140,248], // 238
        [249,135],[250, 69],[ 80,251],[140,252],[249,135],[250, 69],[ 80,251], // 245
        [140,252],[  0,  0],[  0,  0],[  0,  0]]; // 252

        STATE_TABLE[state][sel]
    }

    fn squash(d: i32) -> i32 {
        const T: [i32; 33] = [
            1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
            3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
        ];

        if d > 2047 {
            return 4095;
        }
        if d < -2047 {
            return 0;
        }

        let w = d & 127;
        let d = ((d >> 7) + 16) as usize;
        (T[d] * (128 - w) + T[d + 1] * w + 64) >> 7
    }

    static mut STRETCH_T: [i16; 4096] = [0; 4096];
    static mut DT: [i32; 1024] = [0; 1024];
    static mut HAVE_INIT_TABLES: bool = false;

    fn init_tables() {
        unsafe {
            if !HAVE_INIT_TABLES {
                // init dt
                for i in 0..1024 {
                    DT[i] = (16384 / (i + i + 3)) as i32;
                }

                // init stretch
                let mut pi = 0;
                // Invert squash()
                for x in -2047..=2047 {
                    let i = squash(x);
                    for j in pi..=i {
                        STRETCH_T[j as usize] = x as i16;
                    }
                    pi = i + 1;
                }
                STRETCH_T[4095] = 2047;
            }
            HAVE_INIT_TABLES = true;
        }
    }

    fn stretch(p: usize) -> i16 {
        debug_assert!(p < 4096);
        unsafe { STRETCH_T[p] }
    }

    struct StateMap {
        n: usize,    // Number of contexts
        cxt: usize,  // Context of last prediction
        t: Vec<u32>, // cxt -> prediction in high 22 bits, count in low 10 bits
    }

    impl StateMap {
        fn new(n: usize) -> Self {
            let t = vec![1u32 << 31; n];
            Self { n, cxt: 0, t }
        }

        /// Update bit y (0..1), predict next bit in context cx
        /// Returns probability (0..4095) that the next y=1
        /// limit (1..1023, default 1023) is the maximum count for computing a prediction
        fn p(&mut self, y: i32, cx: usize, limit: i32) -> i32 {
            debug_assert!(y == 0 || y == 1, "y must be 0 or 1");
            debug_assert!(cx < self.n, "cx must be in range 0..{}", self.n - 1);
            debug_assert!(limit > 0 && limit < 1024, "limit must be 1..1023");

            self.update(y, limit);
            self.cxt = cx;
            (self.t[self.cxt] >> 20) as i32
        }

        fn update(&mut self, y: i32, limit: i32) {
            debug_assert!(self.cxt < self.n, "cxt must be in range 0..{}", self.n - 1);

            let val = self.t[self.cxt];
            let n = (val & 1023) as i32; // count in low 10 bits
            let p = (val >> 10) as i32; // prediction in high 22 bits

            // Update count, but don't exceed limit
            let new_n = if n < limit { n + 1 } else { limit };

            // Get dt value safely
            let dt_val = unsafe {
                debug_assert!(n >= 0 && n < 1024, "n must be in range 0..1023");
                DT[n as usize]
            };

            // Update prediction
            let update = (((y << 22) - p) >> 3) * dt_val;
            let new_val = (val & 0xFFFF_FC00) | (new_n as u32 & 1023);
            self.t[self.cxt] = new_val + ((update as u32) & 0xFFFF_FC00);
        }
    }

    struct APM {
        state_map: StateMap,
    }

    impl APM {
        /// Creates an APM with n contexts using 96*n bytes memory
        fn new(n: usize) -> Self {
            let mut state_map = StateMap::new(n * 24);

            // Initialize the table
            for i in 0..state_map.n {
                let p: i32 = (((i as i32) % 24 * 2 + 1) * 4096) / 48 - 2048;
                state_map.t[i] = ((squash(p) as u32) << 20) + 6;
            }

            Self { state_map }
        }

        /// Update and return a new probability (0..4095)
        /// pr (0..4095) is considered part of the context
        /// y (0..1) is the last bit
        /// cx (0..n-1) is the other context
        /// limit (0..1023) defaults to 255
        fn pp(&mut self, y: i32, pr: i32, cx: usize, limit: i32) -> i32 {
            debug_assert!(y == 0 || y == 1, "y must be 0 or 1");
            debug_assert!(pr >= 0 && pr < 4096, "pr must be 0..4095");
            debug_assert!(
                cx < self.state_map.n / 24,
                "cx must be in range 0..{}",
                self.state_map.n / 24 - 1
            );
            debug_assert!(limit > 0 && limit < 1024, "limit must be 1..1023");

            self.state_map.update(y, limit);

            let pr = (stretch(pr as usize) as i32 + 2048) * 23;
            let wt = (pr & 0xFFF) as i32; // interpolation weight of next element

            let cx_new = cx * 24 + ((pr >> 12) as usize);
            debug_assert!(cx_new < self.state_map.n - 1, "cx_new out of bounds");

            let p1 = (self.state_map.t[cx_new] >> 13) as i32;
            let p2 = (self.state_map.t[cx_new + 1] >> 13) as i32;

            let result = (p1 * (0x1000 - wt) + p2 * wt) >> 19;

            self.state_map.cxt = cx_new + ((wt >> 11) as usize);
            result
        }
    }

    struct Mixer {
        n: usize,     // max inputs
        m: usize,     // max contexts
        tx: Vec<i32>, // N inputs
        wx: Vec<i32>, // N*M weights
        cxt: usize,   // context
        nx: usize,    // Number of inputs in tx, 0 to N
        pr: i32,      // last result (scaled 12 bits)
    }

    impl Mixer {
        fn new(n: usize, m: usize) -> Self {
            debug_assert!(n > 0 && m > 0);

            Mixer {
                n,
                m,
                tx: vec![0; n],
                wx: vec![0; n * m],
                cxt: 0,
                nx: 0,
                pr: 2048, // 0.5 in 12-bit fixed point
            }
        }

        fn update(&mut self, y: i32) {
            let err = ((y << 12) - self.pr) * 7;
            debug_assert!(err >= -32768 && err < 32768);

            let start_idx = self.cxt * self.n;
            Self::train(&self.tx, &mut self.wx[start_idx..start_idx + self.n], err);
            self.nx = 0;
        }

        fn add(&mut self, x: i32) {
            debug_assert!(self.nx < self.n);
            self.tx[self.nx] = x;
            self.nx += 1;
        }

        fn set(&mut self, cx: usize) {
            debug_assert!(cx < self.m);
            self.cxt = cx;
        }

        fn p(&mut self) -> i32 {
            let start_idx = self.cxt * self.n;
            let dot = Self::dot_product(&self.tx, &self.wx[start_idx..start_idx + self.n]);
            self.pr = squash(dot >> 8);
            self.pr
        }

        fn dot_product(t: &[i32], w: &[i32]) -> i32 {
            debug_assert_eq!(t.len(), w.len());
            let sum = t
                .iter()
                .zip(w.iter())
                .map(|(&t_val, &w_val)| t_val * w_val)
                .sum::<i32>();
            sum >> 8
        }

        fn train(t: &[i32], w: &mut [i32], err: i32) {
            debug_assert_eq!(t.len(), w.len());
            for i in 0..t.len() {
                w[i] += (t[i] * err + 0x8000) >> 16;
            }
        }
    }

    struct MatchModel {
        n: usize,         // buffer size (n/2-1)
        hn: usize,        // hash table size (n/8-1)
        buf: Vec<u8>,     // input buffer
        ht: Vec<usize>,   // context hash -> next byte in buf
        pos: usize,       // number of bytes in buf
        match_pos: usize, // pointer to current byte in matched context in buf
        len: usize,       // length of match
        h1: usize,        // context hash 1
        h2: usize,        // context hash 2
        c0: u8,           // last 0-7 bits of y
        bcount: usize,    // number of bits in c0 (0..7)
        sm: StateMap,     // len, bit, last byte -> prediction
    }

    impl MatchModel {
        const MAXLEN: usize = 62; // maximum match length, at most 62

        /// Creates a MatchModel using n bytes of memory.
        fn new(n: usize) -> Self {
            debug_assert!(
                n >= 8 && (n & (n - 1)) == 0,
                "n must be a power of 2 at least 8"
            );

            let n = n / 2 - 1;
            let hn = n / 8 - 1;
            let buf = vec![0u8; n + 1];
            let ht = vec![0usize; hn + 1];
            Self {
                n,
                hn,
                buf,
                ht,
                pos: 0,
                match_pos: 0,
                len: 0,
                h1: 0,
                h2: 0,
                c0: 1,
                bcount: 0,
                sm: StateMap::new(56 << 8), // 56 * 256 contexts
            }
        }

        /// Update bit y (0..1) and predict next bit to Mixer m.
        /// Returns the length of context matched (0..62).
        fn p(&mut self, y: i32, m: &mut Mixer) -> usize {
            // update context
            self.c0 = self.c0.wrapping_add(self.c0) + (y as u8);
            self.bcount += 1;
            if self.bcount == 8 {
                self.bcount = 0;

                // Update hashes
                self.h1 = (self.h1 * (3 << 3) + self.c0 as usize) & self.hn;
                self.h2 = (self.h2 * (5 << 5) + self.c0 as usize) & self.hn;

                // Store byte in buffer
                self.buf[self.pos] = self.c0;
                self.pos += 1;
                self.c0 = 1;
                self.pos &= self.n;

                // find or extend match
                if self.len > 0 {
                    self.match_pos += 1;
                    self.match_pos &= self.n;
                    if self.len < Self::MAXLEN {
                        self.len += 1;
                    }
                } else {
                    self.match_pos = self.ht[self.h1];
                    if self.match_pos != self.pos {
                        let mut i =
                            (self.match_pos as isize - self.len as isize - 1) & self.n as isize;
                        while self.len < Self::MAXLEN
                            && i as usize != self.pos
                            && self.buf[i as usize]
                                == self.buf
                                    [(self.pos as isize - self.len as isize - 1) as usize & self.n]
                        {
                            self.len += 1;
                            i = (self.match_pos as isize - self.len as isize - 1) & self.n as isize;
                        }
                    }
                }

                if self.len < 2 {
                    self.len = 0;
                    self.match_pos = self.ht[self.h2];
                    if self.match_pos != self.pos {
                        let mut i =
                            (self.match_pos as isize - self.len as isize - 1) & self.n as isize;
                        while self.len < Self::MAXLEN
                            && i as usize != self.pos
                            && self.buf[i as usize]
                                == self.buf
                                    [(self.pos as isize - self.len as isize - 1) as usize & self.n]
                        {
                            self.len += 1;
                            i = (self.match_pos as isize - self.len as isize - 1) & self.n as isize;
                        }
                    }
                }
            }

            // predict
            let mut cxt = self.c0 as usize;
            if self.len > 0
                && ((self.buf[self.match_pos] as usize + 256) >> (8 - self.bcount))
                    == self.c0 as usize
            {
                let b = (self.buf[self.match_pos] >> (7 - self.bcount)) & 1; // next bit
                let b = b as usize;

                if self.len < 16 {
                    cxt = self.len * 2 + b;
                } else {
                    cxt = (self.len >> 2) * 2 + b + 24;
                }
                cxt = cxt * 256 + self.buf[(self.pos as isize - 1) as usize & self.n] as usize;
            } else {
                self.len = 0;
            }
            m.add(stretch(self.sm.p(y, cxt, 1023) as usize) as i32);

            // update index
            if self.bcount == 0 {
                self.ht[self.h1] = self.pos;
                self.ht[self.h2] = self.pos;
            }
            self.len
        }
    }

    /// A HashTable maps a 32-bit index to an array of B bytes.
    /// Layout: [Checksum (1 byte), Priority (1 byte), Data (B-2 bytes)]
    struct HashTable<const B: usize> {
        t: Vec<u8>,
        n: usize, // Size in bytes
    }

    impl<const B: usize> HashTable<B> {
        fn new(n: usize) -> Self {
            debug_assert!(B >= 2 && (B & (B - 1)) == 0, "B must be power of 2 >= 2");
            debug_assert!(
                n >= B * 4 && (n & (n - 1)) == 0,
                "n must be power of 2 >= B*4"
            );

            let t = vec![0u8; n + B * 4 + 64];
            Self { t, n }
        }

        // Corresponds to operator[] in cpp but returns index rather than
        // pointer.
        fn get_block_index(&mut self, mut i: u32) -> usize {
            i *= 123456791;
            i = (i << 16) | (i >> 16);
            i *= 234567891;
            let chk = (i >> 24) as u8;
            let mut i = (i as usize).wrapping_mul(B) & (self.n - B);

            if self.t[i] == chk {
                return i;
            }
            if self.t[i ^ B] == chk {
                return i ^ B;
            }
            if self.t[i ^ B * 2] == chk {
                return i ^ B * 2;
            }
            if self.t[i + 1] > self.t[i + 1 ^ B] || self.t[i + 1] > self.t[i + 1 ^ B * 2] {
                i ^= B;
            }
            if self.t[i + 1] > self.t[i + 1 ^ B ^ B * 2] {
                i ^= B ^ B * 2;
            }
            self.t[i..i + B].fill(0);
            self.t[i] = chk;
            i
        }

        /// Helper to access data directly
        fn get_mut(&mut self, index: usize) -> &mut u8 {
            &mut self.t[index]
        }

        fn get(&self, index: usize) -> u8 {
            self.t[index]
        }
    }

    struct Predictor {
        pr: i32,              // next prediction
        t0: Box<[u8; 65536]>, // order 1 cxt -> state (on heap to avoid stack overflow)
        t: HashTable<16>,     // cxt -> state
        c0: u32,              // last 0-7 bits with leading 1
        c4: u32,              // last 4 bytes

        // Pointers to bit history (or in our case indices into t0 or t)
        // cp[0] is index into t0
        // cp[1..5] are indices into t
        //
        // we distinguish and use cp0 for cp[0] instead.
        cp0: usize,
        cp: [usize; 6], // cp[0] is unused here to match 1-5 indexing convenience

        bcount: usize,     // bit count
        sm: Vec<StateMap>, // StateMaps (using Vec to hold the 6 maps)
        a1: APM,
        a2: APM,
        h: [u32; 6],
        m: Mixer,
        mm: MatchModel,
    }

    impl Predictor {
        fn new() -> Self {
            let mut sm = Vec::with_capacity(6);
            for _ in 0..6 {
                sm.push(StateMap::new(256));
            }

            // Initialize cp pointers
            // t0 is initialized to 0 (default for u8)
            let t0 = Box::new([0u8; 65536]);
            let t = HashTable::<16>::new(MEM * 2);

            // Initial setup matching C++ static initialization
            let cp0 = 0; // Will be set in update
            let cp = [0; 6]; // Will be set in update

            // Initialize history hashes
            let h = [0; 6];

            Self {
                pr: 2048,
                t0,
                t,
                c0: 1,
                c4: 0,
                cp0,
                cp,
                bcount: 0,
                sm,
                a1: APM::new(0x100),
                a2: APM::new(0x4000),
                h,
                m: Mixer::new(7, 80),
                mm: MatchModel::new(MEM),
            }
        }

        fn p(&self) -> i32 {
            debug_assert!(self.pr >= 0 && self.pr < 4096);
            self.pr
        }

        fn update(&mut self, y: u32) {
            debug_assert!(y == 0 || y == 1);

            // update model
            self.t0[self.cp0] = nex(self.t0[self.cp0] as usize, y as usize);
            for k in 1..6 {
                let i = self.cp[k];
                *self.t.get_mut(i) = nex(self.t.get(i) as usize, y as usize);
            }
            self.m.update(y as i32);

            // update context
            self.bcount += 1;
            self.c0 += self.c0 + y;
            if self.c0 >= 256 {
                self.c0 -= 256;
                self.c4 = self.c4 << 8 | self.c0;

                self.h[0] = self.c0 << 8;
                self.h[1] = (self.c4 & 0xffff) << 5 | 0x57000000;
                self.h[2] = (self.c4 << 8) * 3;
                self.h[3] = self.c4 * 5;
                self.h[4] = self.h[4] * (11 << 5) + self.c0 * 13 & 0x3fffffff;
                if self.c0 >= 65 && self.c0 <= 90 {
                    self.c0 += 32;
                }
                if self.c0 >= 97 && self.c0 <= 122 {
                    self.h[5] = (self.h[5] + self.c0) * (7 << 3);
                } else {
                    self.h[5] = 0;
                }
                self.cp[1] = self.t.get_block_index(self.h[1]) + 1;
                self.cp[2] = self.t.get_block_index(self.h[2]) + 1;
                self.cp[3] = self.t.get_block_index(self.h[3]) + 1;
                self.cp[4] = self.t.get_block_index(self.h[4]) + 1;
                self.cp[5] = self.t.get_block_index(self.h[5]) + 1;
                self.c0 = 1;
                self.bcount = 0;
            }

            if self.bcount == 4 {
                self.cp[1] = self.t.get_block_index(self.h[1] + self.c0) + 1;
                self.cp[2] = self.t.get_block_index(self.h[2] + self.c0) + 1;
                self.cp[3] = self.t.get_block_index(self.h[3] + self.c0) + 1;
                self.cp[4] = self.t.get_block_index(self.h[4] + self.c0) + 1;
                self.cp[5] = self.t.get_block_index(self.h[5] + self.c0) + 1;
            } else if self.bcount > 0 {
                let j = y + 1 << (self.bcount & 3) - 1;
                for k in 1..6 {
                    self.cp[k] += j as usize;
                }
            }
            self.cp0 = self.h[0] as usize + self.c0 as usize;

            // predict
            let len = self.mm.p(y as i32, &mut self.m);
            let mut order = 0;
            if len == 0 {
                if self.t.get(self.cp[4]) != 0 {
                    order += 1;
                }
                if self.t.get(self.cp[3]) != 0 {
                    order += 1;
                }
                if self.t.get(self.cp[2]) != 0 {
                    order += 1;
                }
                if self.t.get(self.cp[1]) != 0 {
                    order += 1;
                }
            } else {
                order = 5
                    + (if len >= 8 { 1 } else { 0 })
                    + (if len >= 12 { 1 } else { 0 })
                    + (if len >= 16 { 1 } else { 0 })
                    + (if len >= 32 { 1 } else { 0 });
            }
            self.m.add(
                stretch(self.sm[0].p(y as i32, self.t0[self.cp0] as usize, 1023) as usize) as i32,
            );
            for k in 1..6 {
                let val = self.t.get(self.cp[k]);
                let p_val = self.sm[k].p(y as i32, val as usize, 1023);
                self.m.add(stretch(p_val as usize) as i32);
            }
            self.m.set(order + 10 * ((self.h[0] >> 13) as usize));
            self.pr = self.m.p();
            self.pr = self.pr + 3 * self.a1.pp(y as i32, self.pr, self.c0 as usize, 1023) >> 2;
            self.pr = self.pr
                + 3 * self.a2.pp(
                    y as i32,
                    self.pr,
                    (self.c0 as usize) ^ (self.h[0] as usize >> 2),
                    1023,
                )
                >> 2;
        }
    }

    const MEM: usize = 1 << (9 + 20);

    #[derive(Copy, Clone, PartialEq, Debug)]
    enum Mode {
        Compress,
        Decompress,
    }

    struct Encoder {
        predictor: Predictor,
        mode: Mode,
        data: Cursor<Vec<u8>>,
        output_buffer: Vec<u8>,
        x1: u32, // Range low, scaled by 2^32
        x2: u32, // Range high, scaled by 2^32
        x: u32,  // For decompress: last 4 input bytes
    }

    impl Encoder {
        fn new(mode: Mode, data: Vec<u8>) -> IoResult<Self> {
            init_tables();
            let mut encoder = Self {
                predictor: Predictor::new(),
                mode,
                data: Cursor::new(data),
                output_buffer: Vec::new(),
                x1: 0,
                x2: 0xffffffff,
                x: 0,
            };

            if mode == Mode::Decompress {
                let mut buf = [0u8; 4];
                encoder.data.read_exact(&mut buf)?;
                encoder.x = u32::from_be_bytes(buf);
            }
            Ok(encoder)
        }

        /// Compress/decompress one bit.
        /// In compress mode, `y` is the bit to encode.
        /// In decompress mode, `y` is unused and returns the decoded bit.
        fn code(&mut self, y: Option<i32>) -> IoResult<i32> {
            let p = self.predictor.p();
            debug_assert!(p >= 0 && p < 4096);
            let p = if p < 2048 { p + 1 } else { p };

            // Calculate: xmid = x1 + ((x2 - x1) >> 12) * p + (((x2 - x1) & 0xfff) * p >> 12)
            let range = self.x2.wrapping_sub(self.x1);
            let range_low = range >> 12;
            let range_high = range & 0xfff;
            let term1 = range_low.wrapping_mul(p as u32);
            let term2 = (range_high.wrapping_mul(p as u32)) >> 12;
            let xmid = self.x1.wrapping_add(term1).wrapping_add(term2);

            debug_assert!(xmid >= self.x1 && xmid < self.x2);

            let bit_y = match self.mode {
                Mode::Compress => y.expect("Must provide bit in compress mode"),
                Mode::Decompress => {
                    if self.x <= xmid {
                        1
                    } else {
                        0
                    }
                }
            };

            // Update range based on bit
            if bit_y != 0 {
                self.x2 = xmid;
            } else {
                self.x1 = xmid.wrapping_add(1);
            }

            // Update predictor
            self.predictor.update(bit_y as u32);

            // Normalize range: shift out equal leading bytes
            while (self.x1 ^ self.x2) & 0xff000000 == 0 {
                if self.mode == Mode::Compress {
                    self.output_buffer.push((self.x2 >> 24) as u8);
                }
                self.x1 <<= 8;
                self.x2 = (self.x2 << 8) + 0xff;
                if self.mode == Mode::Decompress {
                    let mut buf = [0u8; 1];
                    let byte = if self.data.read_exact(&mut buf).is_ok() {
                        buf[0]
                    } else {
                        255 // Match C++ EOF behavior
                    };
                    self.x = (self.x << 8).wrapping_add(byte as u32);
                }
            }

            Ok(bit_y)
        }

        /// Compress one byte
        fn compress(&mut self, c: u8) -> IoResult<()> {
            assert_eq!(self.mode, Mode::Compress);
            for i in (0..8).rev() {
                let bit = ((c >> i) & 1) as i32;
                self.code(Some(bit))?;
            }
            Ok(())
        }

        /// Decompress and return one byte
        fn decompress(&mut self) -> IoResult<u8> {
            assert_eq!(self.mode, Mode::Decompress);
            let mut c = 0u8;
            for _ in 0..8 {
                let bit = self.code(None)?;
                c = (c << 1) | (bit as u8);
            }
            Ok(c)
        }

        fn flush(&mut self) -> IoResult<()> {
            if self.mode == Mode::Compress {
                self.output_buffer.push((self.x1 >> 24) as u8);
            }
            Ok(())
        }

        fn into_output(self) -> Vec<u8> {
            self.output_buffer
        }
    }

    pub fn compress_text(data: &[u8]) -> Vec<u8> {
        // Compress first
        let mut encoder = Encoder::new(Mode::Compress, Vec::new()).unwrap();
        for &byte in data {
            encoder.compress(byte).unwrap();
        }
        encoder.flush().unwrap();

        // Prepend header to compressed data
        let mut output = encoder.into_output();
        let size = data.len() as u32;

        // Create header + compressed data
        let mut final_output = Vec::with_capacity(output.len() + 4);
        final_output.extend_from_slice(&size.to_le_bytes());
        final_output.append(&mut output);

        final_output
    }

    pub fn decompress_text(data: &[u8]) -> Vec<u8> {
        if data.len() < 4 {
            return Vec::new();
        }

        // Extract header using array slice conversion
        let header: [u8; 4] = data[0..4].try_into().expect("Invalid header");
        let original_size = u32::from_le_bytes(header) as usize;

        // Remaining bytes are compressed data
        let compressed_data = &data[4..];

        let mut decoder = Encoder::new(Mode::Decompress, compressed_data.to_vec()).unwrap();
        let mut decompressed = Vec::with_capacity(original_size);

        for _ in 0..original_size {
            decompressed.push(decoder.decompress().unwrap());
        }

        decompressed
    }
}

pub struct KjcaoCodec;

fn preprocess_repo_names(input_data: &[u8]) -> Vec<u8> {
    // 1. Parse input data
    let input_str = String::from_utf8_lossy(input_data);
    let lines: Vec<&str> = input_str
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();

    let mut owners = Vec::with_capacity(lines.len());
    let mut repos = Vec::with_capacity(lines.len());

    for line in lines {
        if let Some(pos) = line.find('/') {
            let (owner, repo) = line.split_at(pos);
            // Skip the '/'
            owners.push(owner);
            repos.push(&repo[1..]);
        } else {
            // Handle edge case of no slash (rare but possible in dirty data)
            owners.push("");
            repos.push(line);
        }
    }

    // 2. Process Owners (Dictionary Encoding)
    // Find unique owners and sort them by frequency to give smaller IDs to common owners
    let mut owner_counts = HashMap::new();
    for owner in &owners {
        *owner_counts.entry(owner).or_insert(0) += 1;
    }

    // Create vector of (owner, count) and sort by frequency (descending), then alphabetically
    let mut owner_freq: Vec<(&str, i32)> = owner_counts.into_iter().map(|(k, v)| (*k, v)).collect();
    owner_freq.sort_by(|a, b| {
        // Primary sort: frequency descending
        b.1.cmp(&a.1)
            // Secondary sort: name ascending
            .then_with(|| a.0.cmp(&b.0))
    });

    let sorted_owners: Vec<&str> = owner_freq.iter().map(|(owner, _)| *owner).collect();
    let owner_to_id: HashMap<&str, u32> = sorted_owners
        .iter()
        .enumerate()
        .map(|(i, &owner)| (owner, i as u32))
        .collect();

    // 3. Create output buffer
    let mut output = Vec::new();
    let mut cursor = Cursor::new(&mut output);

    // 4. Write the Header (The Dictionary)
    // Write count of unique owners (4 bytes, little-endian)
    cursor
        .write_all(&(sorted_owners.len() as u32).to_le_bytes())
        .unwrap();

    // Write the owner strings
    for owner in &sorted_owners {
        cursor.write_all(owner.as_bytes()).unwrap();
        cursor.write_all(&[0]).unwrap(); // Null terminator
    }

    // 5. Encode the Data Stream
    // Block A: Owner IDs using simple varint encoding
    let mut encoded_owner_ids = Vec::new();
    for owner in &owners {
        let oid = owner_to_id[owner];
        let mut n = oid;

        // Simple Varint encoding
        while n >= 128 {
            encoded_owner_ids.push(((n & 0x7f) | 0x80) as u8);
            n >>= 7;
        }
        encoded_owner_ids.push(n as u8);
    }

    // Write size of owner IDs block (4 bytes, little-endian)
    cursor
        .write_all(&(encoded_owner_ids.len() as u32).to_le_bytes())
        .unwrap();
    cursor.write_all(&encoded_owner_ids).unwrap();

    // Block B: Repo Names with Transformations
    for i in 0..owners.len() {
        let owner = owners[i];
        let repo = repos[i];

        // Transformation 1: Exact Match
        if repo == owner {
            cursor.write_all(&[0x01]).unwrap(); // Special byte: Same as owner
            continue;
        }

        // Transformation 2: Prefix Match (e.g., "ruby-git/ruby-git")
        if repo.starts_with(owner) {
            let remainder = &repo[owner.len()..];

            // Check if remainder starts with typical separators
            if let Some(first_char) = remainder.chars().next() {
                if matches!(first_char, '-' | '_' | '.') {
                    cursor.write_all(&[0x02]).unwrap(); // Special byte: Starts with owner
                    cursor.write_all(remainder.as_bytes()).unwrap();
                    cursor.write_all(&[0]).unwrap(); // End of string
                    continue;
                }
            }
        }

        // Transformation 3: Common Prefix stripping (naive)
        // If not optimized above, just write raw
        cursor.write_all(&[0x00]).unwrap(); // Raw string flag
        cursor.write_all(repo.as_bytes()).unwrap();
        cursor.write_all(&[0]).unwrap(); // End of string
    }

    output
}

fn invert_preprocess_repo_name(compressed_data: &[u8]) -> Vec<u8> {
    let mut cursor = compressed_data;
    let mut result = Vec::new();

    // Helper function to read bytes from cursor
    fn read_u32_le(data: &mut &[u8]) -> u32 {
        if data.len() < 4 {
            return 0;
        }
        let bytes = [data[0], data[1], data[2], data[3]];
        *data = &data[4..];
        u32::from_le_bytes(bytes)
    }

    // 1. Read owner dictionary count
    if cursor.len() < 4 {
        return result;
    }
    let owner_count = read_u32_le(&mut cursor) as usize;

    // 2. Read owner dictionary
    let mut owners = Vec::with_capacity(owner_count);
    for _ in 0..owner_count {
        // Find null terminator
        let null_pos = cursor.iter().position(|&b| b == 0).unwrap_or(cursor.len());
        if null_pos >= cursor.len() {
            break;
        }
        let owner_bytes = &cursor[..null_pos];
        let owner = String::from_utf8_lossy(owner_bytes);
        owners.push(owner.to_string());
        cursor = &cursor[null_pos + 1..]; // Skip null terminator
    }

    // 3. Read owner IDs block size
    if cursor.len() < 4 {
        return result;
    }
    let owner_ids_size = read_u32_le(&mut cursor) as usize;

    if cursor.len() < owner_ids_size {
        return result;
    }

    // 4. Decode varint-encoded owner IDs
    let owner_ids_data = &cursor[..owner_ids_size];
    cursor = &cursor[owner_ids_size..];

    let mut owner_ids = Vec::new();
    let mut pos = 0;
    while pos < owner_ids_data.len() {
        let mut id: u32 = 0;
        let mut shift = 0;

        loop {
            if pos >= owner_ids_data.len() {
                break;
            }
            let byte = owner_ids_data[pos];
            pos += 1;

            id |= ((byte & 0x7F) as u32) << shift;

            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
        }

        owner_ids.push(id as usize);
    }

    // 5. Decode repo names
    let mut repos = Vec::with_capacity(owner_ids.len());
    let mut repo_data = cursor;

    while repos.len() < owner_ids.len() && !repo_data.is_empty() {
        let flag = repo_data[0];
        repo_data = &repo_data[1..];

        match flag {
            0x01 => {
                // Same as owner
                let owner_idx = owner_ids[repos.len()];
                if owner_idx < owners.len() {
                    repos.push(owners[owner_idx].clone());
                } else {
                    repos.push(String::new());
                }
            }
            0x02 => {
                // Starts with owner + separator + remainder
                let owner_idx = owner_ids[repos.len()];
                let owner = if owner_idx < owners.len() {
                    &owners[owner_idx]
                } else {
                    ""
                };

                // Find null terminator for remainder
                let null_pos = repo_data
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(repo_data.len());
                if null_pos >= repo_data.len() {
                    break;
                }

                let remainder = &repo_data[..null_pos];
                let remainder_str = String::from_utf8_lossy(remainder);
                repo_data = &repo_data[null_pos + 1..]; // Skip null terminator

                repos.push(format!("{}{}", owner, remainder_str));
            }
            0x00 => {
                // Raw string
                let null_pos = repo_data
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(repo_data.len());
                if null_pos >= repo_data.len() {
                    break;
                }

                let repo_bytes = &repo_data[..null_pos];
                let repo = String::from_utf8_lossy(repo_bytes);
                repo_data = &repo_data[null_pos + 1..]; // Skip null terminator

                repos.push(repo.to_string());
            }
            _ => {
                // Unknown flag, treat as error
                repos.push(String::new());
            }
        }
    }

    // 6. Combine owners and repos
    for i in 0..owner_ids.len().min(repos.len()) {
        let owner_idx = owner_ids[i];
        if owner_idx < owners.len() {
            let owner = &owners[owner_idx];
            let repo = &repos[i];

            if !owner.is_empty() || !repo.is_empty() {
                result.extend_from_slice(format!("{}/{}\n", owner, repo).as_bytes());
            } else {
                result.extend_from_slice(b"\n");
            }
        }
    }

    if result.last() == Some(&b'\n') {
        result.pop();
    }
    result
}

fn parse_timestamp(ts: &str) -> Result<i64, Box<dyn Error>> {
    Ok(DateTime::parse_from_rfc3339(ts)?.timestamp())
}

fn format_timestamp(epoch: i64) -> String {
    Utc.timestamp_opt(epoch, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_default()
}

impl KjcaoCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for KjcaoCodec {
    fn name(&self) -> &str {
        "kjcao"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut repo_pairs: HashSet<(u64, String)> = HashSet::new();
        let mut type_set: HashSet<String> = HashSet::new();

        #[derive(Clone)]
        struct RawEvent {
            id: u64,
            type_name: String,
            repo_id: u64,
            repo_name: String,
            ts: i64,
        }

        let mut raw_events: Vec<RawEvent> = events
            .iter()
            .map(|(key, value)| {
                let id = key.id.parse::<u64>().unwrap();
                let repo_id = value.repo.id;
                let repo_name = value.repo.name.clone();
                let ts = parse_timestamp(&value.created_at).unwrap();

                repo_pairs.insert((repo_id, repo_name.clone()));
                type_set.insert(key.event_type.clone());

                RawEvent {
                    id,
                    type_name: key.event_type.clone(),
                    repo_id,
                    repo_name,
                    ts,
                }
            })
            .collect();

        // Sort mapping by repo_id for delta compression
        let mut mapping_entries: Vec<(u64, String)> = repo_pairs.into_iter().collect();
        mapping_entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let num_mapping = mapping_entries.len();
        if num_mapping > (1 << 24) - 1 {
            return Err(format!(
                "Too many unique repo entries: {} (max 16,777,215)",
                num_mapping
            )
            .into());
        }

        let pair_to_idx: HashMap<(u64, String), u32> = mapping_entries
            .iter()
            .enumerate()
            .map(|(idx, (id, name))| ((*id, name.clone()), idx as u32))
            .collect();

        // Sort events by ID for optimal compression
        raw_events.sort_by_key(|e| e.id);
        let num_events = raw_events.len();

        // Build type dictionary (max 16 types)
        let mut type_list: Vec<String> = type_set.into_iter().collect();
        type_list.sort();
        if type_list.len() > 16 {
            return Err(format!("Too many event types: {} (max 16)", type_list.len()).into());
        }
        let type_to_idx: HashMap<String, u8> = type_list
            .iter()
            .enumerate()
            .map(|(idx, t)| (t.clone(), idx as u8))
            .collect();

        // 1. Mapping IDs: sorted u64s (pco will delta encode)
        let mapping_ids: Vec<u64> = mapping_entries.iter().map(|(id, _)| *id).collect();

        // 2. Mapping names: newline-separated UTF-8
        let mapping_names_raw: Vec<u8> = mapping_entries
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();

        // 3. Event type dict: newline-separated
        let dict_raw: Vec<u8> = type_list.join("\n").into_bytes();

        // 4. Event type indices: Vec<u8>
        let type_indices: Vec<u8> = raw_events
            .iter()
            .map(|e| *type_to_idx.get(&e.type_name).unwrap())
            .collect();

        // 5. Event IDs: sorted u64s (pco will delta encode)
        let event_ids: Vec<u64> = raw_events.iter().map(|e| e.id).collect();

        // 6. Repo indices: Vec<u32>
        let repo_indices: Vec<u32> = raw_events
            .iter()
            .map(|e| *pair_to_idx.get(&(e.repo_id, e.repo_name.clone())).unwrap())
            .collect();

        // 7. Timestamps: i64s (pco will delta encode)
        let timestamps: Vec<i64> = raw_events.iter().map(|e| e.ts).collect();

        // --- Compress & Assemble ---
        let mut pco_config = ChunkConfig::default();
        pco_config.compression_level = 12;
        pco_config.paging_spec = EqualPagesUpTo(1 << 18);

        // Use pcodec for numeric columns with clear deltas. Use lpaq1 for
        // everything else.
        let mapping_ids_comp = simple_compress(&mapping_ids, &pco_config)?;
        let type_comp = lpaq1::compress_text(&type_indices);
        let event_ids_comp = simple_compress(&event_ids, &pco_config)?;
        let repo_comp = lpaq1::compress_text(
            repo_indices
                .iter()
                .flat_map(|&num| num.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let ts_comp = simple_compress(&timestamps, &pco_config)?;
        let mapping_names_comp =
            lpaq1::compress_text(preprocess_repo_names(mapping_names_raw.as_slice()).as_slice());
        let dict_comp = lpaq1::compress_text(&dict_raw);

        // --- Header (40 bytes) ---
        let mut header = Vec::with_capacity(40);
        header.write_all(&(num_events as u32).to_le_bytes())?;
        header.write_all(&(num_mapping as u32).to_le_bytes())?;
        header.write_all(&(dict_raw.len() as u16).to_le_bytes())?;
        header.push(type_list.len() as u8);
        header.push(0); // padding
        header.write_all(&(mapping_ids_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(mapping_names_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(dict_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(type_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(event_ids_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(repo_comp.len() as u32).to_le_bytes())?;
        header.write_all(&(ts_comp.len() as u32).to_le_bytes())?;

        let mut output = header;
        output.extend_from_slice(&mapping_ids_comp);
        output.extend_from_slice(&mapping_names_comp);
        output.extend_from_slice(&dict_comp);
        output.extend_from_slice(&type_comp);
        output.extend_from_slice(&event_ids_comp);
        output.extend_from_slice(&repo_comp);
        output.extend_from_slice(&ts_comp);

        Ok(Bytes::from(output))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        if bytes.len() < 40 {
            return Err("Input too small for header".into());
        }

        let header = &bytes[..40];

        // Parse header fields (little-endian)
        let num_events = u32::from_le_bytes(header[0..4].try_into()?) as usize;
        let num_mapping = u32::from_le_bytes(header[4..8].try_into()?) as usize;
        let dict_size = u16::from_le_bytes(header[8..10].try_into()?) as usize;
        let num_types = header[10] as usize;
        // header[11] is padding

        // Compressed column sizes
        let map_ids_sz = u32::from_le_bytes(header[12..16].try_into()?) as usize;
        let map_names_sz = u32::from_le_bytes(header[16..20].try_into()?) as usize;
        let dict_sz = u32::from_le_bytes(header[20..24].try_into()?) as usize;
        let type_sz = u32::from_le_bytes(header[24..28].try_into()?) as usize;
        let event_ids_sz = u32::from_le_bytes(header[28..32].try_into()?) as usize;
        let repo_sz = u32::from_le_bytes(header[32..36].try_into()?) as usize;
        let ts_sz = u32::from_le_bytes(header[36..40].try_into()?) as usize;

        // Read compressed columns
        let mut offset = 40;
        let mut next_chunk = |size: usize| -> Result<&[u8], Box<dyn Error>> {
            if bytes.len() < offset + size {
                return Err("Truncated input".into());
            }
            let chunk = &bytes[offset..offset + size];
            offset += size;
            Ok(chunk)
        };

        let mapping_ids_comp = next_chunk(map_ids_sz)?;
        let mapping_names_comp = next_chunk(map_names_sz)?;
        let dict_comp = next_chunk(dict_sz)?;
        let type_comp = next_chunk(type_sz)?;
        let event_ids_comp = next_chunk(event_ids_sz)?;
        let repo_comp = next_chunk(repo_sz)?;
        let ts_comp = next_chunk(ts_sz)?;

        let mapping_names_raw =
            invert_preprocess_repo_name(lpaq1::decompress_text(mapping_names_comp).as_slice());
        let dict_raw = lpaq1::decompress_text(dict_comp);

        if dict_raw.len() != dict_size {
            return Err("Event type dictionary size mismatch".into());
        }

        // Mapping table
        let repo_ids: Vec<u64> = simple_decompress(mapping_ids_comp)?;
        if repo_ids.len() != num_mapping {
            return Err("Mapping ID count mismatch".into());
        }

        let mapping_names = String::from_utf8(mapping_names_raw)?;
        let mapping_names: Vec<&str> = mapping_names.split('\n').collect();
        if mapping_names.len() != num_mapping {
            return Err("Mapping name count mismatch".into());
        }

        let mapping: Vec<(u64, &str)> = repo_ids
            .into_iter()
            .zip(mapping_names.into_iter())
            .collect();

        // Event types
        let type_list_str = String::from_utf8(dict_raw)?;
        let type_list: Vec<&str> = type_list_str.split('\n').collect();
        if type_list.len() != num_types {
            return Err("Event type count mismatch".into());
        }

        let type_indices: Vec<u8> = lpaq1::decompress_text(type_comp);
        if type_indices.len() != num_events {
            return Err("Type indices count mismatch".into());
        }

        // Event IDs (already reconstructed by pco)
        let event_ids: Vec<u64> = simple_decompress(event_ids_comp)?;
        if event_ids.len() != num_events {
            return Err("Event ID count mismatch".into());
        }

        // Repo indices
        let repo_indices: Vec<u32> = lpaq1::decompress_text(repo_comp)
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        if repo_indices.len() != num_events {
            return Err("Repo indices count mismatch".into());
        }

        // Timestamps (already reconstructed by pco)
        let timestamps: Vec<i64> = simple_decompress(ts_comp)?;
        if timestamps.len() != num_events {
            return Err("Timestamp count mismatch".into());
        }

        // --- Assemble Events ---
        let mut events = Vec::with_capacity(num_events);
        for i in 0..num_events {
            let repo_idx = repo_indices[i] as usize;
            if repo_idx >= mapping.len() {
                return Err(format!("Repo index {} out of bounds", repo_idx).into());
            }
            let (repo_id, repo_name) = mapping[repo_idx];

            let type_idx = type_indices[i] as usize;
            if type_idx >= type_list.len() {
                return Err(format!("Type index {} out of bounds", type_idx).into());
            }
            let event_type = type_list[type_idx].to_string();

            events.push((
                EventKey {
                    id: event_ids[i].to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name.to_string(),
                        url: format!("https://api.github.com/repos/{}", repo_name),
                    },
                    created_at: format_timestamp(timestamps[i]),
                },
            ));
        }

        events.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(events)
    }
}
