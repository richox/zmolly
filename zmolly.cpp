/*
 * Copyright (C) 2015-2016 by Zhang Li <richselian at gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#include <cstdio>
#include <cstdint>

#include <algorithm>
#include <fstream>
#include <string>
#include <array>
#include <bitset>
#include <memory>
#include <thread>

/*******************************************************************************
 * Arithmetic coder
 ******************************************************************************/
static const auto RC_TOP = 1u << 24;
static const auto RC_BOT = 1u << 16;

struct rc_encoder_t {
    std::ostream& m_ostream;
    uint32_t m_low;
    uint32_t m_range;

    rc_encoder_t(std::ostream& ostream):
        m_ostream(ostream),
        m_low(0),
        m_range(-1) {}

    void encode(uint16_t cum, uint16_t frq, uint16_t sum) {
        m_range /= sum;
        m_low += cum * m_range;
        m_range *= frq;
        while ((m_low ^ (m_low + m_range)) < RC_TOP || (m_range < RC_BOT && ((m_range = -m_low & (RC_BOT - 1)), 1))) {
            m_ostream.put(m_low >> 24);
            m_low <<= 8;
            m_range <<= 8;
        }
    }
    void flush() {
        m_ostream.put(m_low >> 24), m_low <<= 8;
        m_ostream.put(m_low >> 24), m_low <<= 8;
        m_ostream.put(m_low >> 24), m_low <<= 8;
        m_ostream.put(m_low >> 24), m_low <<= 8;
    }
};

struct rc_decoder_t {
    std::istream& m_istream;
    uint32_t m_low;
    uint32_t m_range;
    uint32_t m_code;

    rc_decoder_t(std::istream& istream): m_istream(istream), m_low(0), m_range(-1), m_code(0) {
        m_code = m_code << 8 | istream.get();
        m_code = m_code << 8 | istream.get();
        m_code = m_code << 8 | istream.get();
        m_code = m_code << 8 | istream.get();
    }

    void decode(uint16_t cum, uint16_t frq) {
        m_low += cum * m_range;
        m_range *= frq;
        while ((m_low ^ (m_low + m_range)) < RC_TOP || (m_range < RC_BOT && ((m_range = -m_low & (RC_BOT - 1)), 1))) {
            m_code = m_code << 8 | (unsigned char) m_istream.get();
            m_range <<= 8;
            m_low <<= 8;
        }
    }
    uint16_t decode_cum(uint16_t sum) {
        m_range /= sum;
        return (m_code - m_low) / m_range;
    }
};

/*******************************************************************************
 * PPM Model
 ******************************************************************************/
static const auto PPM_O4_BUCKET_SIZE = 262144;
static const auto PPM_SEE_SIZE = 131072;

struct symbol_counter_t {
    uint8_t m_sym;
    uint8_t m_frq;
    symbol_counter_t():
        m_sym(0),
        m_frq(0) {}
};

struct bit_model_t {
    uint16_t m_c[2];

    int encode(rc_encoder_t* coder, int c) {
        c == 0
            ? coder->encode(0, m_c[0], m_c[0] + m_c[1])
            : coder->encode(m_c[0], m_c[1], m_c[0] + m_c[1]);
        return c;
    }
    int decode(rc_decoder_t* coder) {
        if (m_c[0] > coder->decode_cum(m_c[0] + m_c[1])) {
            coder->decode(0, m_c[0]);
            return 0;
        } else {
            coder->decode(m_c[0], m_c[1]);
            return 1;
        }
    }
    void update(int c) {
        if ((m_c[c] += 15) > 9000) {
            m_c[0] = (m_c[0] + 1) * 0.9;
            m_c[1] = (m_c[1] + 1) * 0.9;
        }
        return;
    }
};

struct dense_model_t {  // dense model types, use for short context
    uint16_t m_sum;
    uint16_t m_cnt;
    uint16_t m_esc;
    std::array<symbol_counter_t, 256> m_symbols;

    dense_model_t():
        m_sum(0),
        m_cnt(0),
        m_esc(0) {}

    int encode(rc_encoder_t* coder, std::bitset<256>& exclude, int c) {
        auto found = 0;
        auto found_pos = 0;
        auto cum = 0;
        auto frq = 0;
        auto sum = 0;
        auto esc = 0;
        auto recent_frq = m_symbols[0].m_frq & -!exclude[m_symbols[0].m_sym];

        if (!exclude.any()) {
            for (auto i = 0; i < m_cnt; i++) {  // no exclusion
                if (m_symbols[i].m_sym == c) {
                    found_pos = i;
                    found = 1;
                    break;
                }
                cum += m_symbols[i].m_frq;
            }
            sum = m_sum;
        } else {
            for (auto i = 0; i < m_cnt; i++) {
                if (m_symbols[i].m_sym == c) {
                    found_pos = i;
                    found = 1;
                }
                cum += m_symbols[i].m_frq & -(!exclude[m_symbols[i].m_sym] && !found);
                sum += m_symbols[i].m_frq & -(!exclude[m_symbols[i].m_sym]);
            }
        }

        esc = m_esc + !m_esc;
        sum += recent_frq + esc;
        frq = m_symbols[found_pos].m_frq;
        if (found_pos == 0) {
            frq += recent_frq;
        } else {
            std::swap(m_symbols[found_pos], m_symbols[0]);
            cum += recent_frq;
        }

        if (!found) {
            for (auto i = 0; i < m_cnt; i++) {  // do exclude
                exclude[m_symbols[i].m_sym] = 1;
            }
            m_symbols[m_cnt].m_frq = m_symbols[0].m_frq;
            m_symbols[m_cnt].m_sym = m_symbols[0].m_sym;
            m_symbols[0].m_sym = c;
            m_symbols[0].m_frq = 0;
            m_cnt += 1;
            cum = sum - esc;
            frq = esc;
        }
        coder->encode(cum, frq, sum);
        return found;
    }

    int decode(rc_decoder_t* coder, std::bitset<256>& exclude) {
        auto cum = 0;
        auto frq = 0;
        auto sum = 0;
        auto esc = 0;
        auto recent_frq = m_symbols[0].m_frq & -!exclude[m_symbols[0].m_sym];
        auto sym = -1;

        for (auto i = 0; i < m_cnt; i++) {
            sum += m_symbols[i].m_frq & -!exclude[m_symbols[i].m_sym];
        }
        esc = m_esc + !m_esc;
        sum += recent_frq + esc;

        auto decode_cum = coder->decode_cum(sum);
        if (sum - esc <= decode_cum) {
            for (auto i = 0; i < m_cnt; i++) {  // do exclude
                exclude[m_symbols[i].m_sym] = 1;
            }
            m_symbols[m_cnt].m_frq = m_symbols[0].m_frq;
            m_symbols[m_cnt].m_sym = m_symbols[0].m_sym;
            m_symbols[0].m_frq = 0;
            m_cnt += 1;
            cum = sum - esc;
            frq = esc;
        } else {
            auto i = 0;
            if (!exclude.any()) {  // no exclusion
                while (cum + recent_frq + m_symbols[i].m_frq <= decode_cum) {
                    cum += m_symbols[i].m_frq;
                    i++;
                }
            } else {
                while (cum + recent_frq + (m_symbols[i].m_frq & -!exclude[m_symbols[i].m_sym]) <= decode_cum) {
                    cum += m_symbols[i].m_frq & -!exclude[m_symbols[i].m_sym];
                    i++;
                }
            }
            frq = m_symbols[i].m_frq;
            sym = m_symbols[i].m_sym;
            if (i == 0) {
                frq += recent_frq;
            } else {
                std::swap(m_symbols[i], m_symbols[0]);
                cum += recent_frq;
            }
        }
        coder->decode(cum, frq);
        return sym;
    }

    void update(int c) {
        m_symbols[0].m_frq += 1;
        m_symbols[0].m_sym = c;
        m_sum += 1;
        m_esc += (m_symbols[0].m_frq == 1) - (m_symbols[0].m_frq == 2);

        if (m_symbols[0].m_frq > 250) {  // rescale
            auto n = 0;
            m_cnt = 0;
            m_sum = 0;
            m_esc = 0;
            for (auto i = 0; i + n < 256; i++) {
                if ((m_symbols[i].m_frq = m_symbols[i + n].m_frq / 2) > 0) {
                    m_symbols[i].m_sym = m_symbols[i + n].m_sym;
                    m_cnt += 1;
                    m_sum += m_symbols[i].m_frq;
                    m_esc += m_symbols[i].m_frq == 1;
                } else {
                    n++;
                    i--;
                }
            }
            std::fill(m_symbols.begin() + m_cnt, m_symbols.end(), symbol_counter_t());
        }
    }

};

struct sparse_model_t {  // sparse model types, use for long context
    sparse_model_t* m_next;
    uint16_t m_sum;
    uint8_t  m_cnt;
    uint8_t  m_visited;
    uint64_t m_context : 48;
    std::array<symbol_counter_t, 54> m_symbols;  // symbol size = 56: make 128byte struct

    sparse_model_t():
        m_next(nullptr),
        m_sum(0),
        m_cnt(0),
        m_visited(0),
        m_context(0) {}

    int encode(bit_model_t* see, rc_encoder_t* coder, int c, std::bitset<256>& exclude) {
        auto cum = 0;
        auto frq = 0;
        auto found_pos = -1;

        for (auto i = 0; i < m_cnt; i++) {  // search for symbol
            if (m_symbols[i].m_sym == c) {
                found_pos = i;
                break;
            }
            cum += m_symbols[i].m_frq;
        }
        if (found_pos >= 0) {  // found -- bring to front of linked-list
            see->encode(coder, 0);
            see->update(0);
            if (m_cnt != 1) {  // no need to encode binary context
                auto recent_frq = (m_symbols[0].m_frq + 6) / 2;  // recency scaling
                if (found_pos == 0) {
                    frq = m_symbols[found_pos].m_frq + recent_frq;
                } else {
                    frq = m_symbols[found_pos].m_frq;
                    cum += recent_frq;
                    auto tmp_symbol = m_symbols[found_pos];
                    std::copy(&m_symbols[0], &m_symbols[found_pos], &m_symbols[1]);
                    m_symbols[0] = tmp_symbol;
                }
                coder->encode(cum, frq, m_sum + recent_frq);
            }
            return 1;

        } else {  // not found -- create new node for sym
            see->encode(coder, 1);
            see->update(1);
            for (auto i = 0; i < m_cnt; i++) {
                exclude[m_symbols[i].m_sym] = 1;  // exclude o4
            }
            if (m_cnt == m_symbols.size()) {
                m_sum -= m_symbols[m_cnt - 1].m_frq;
            } else {
                m_cnt += 1;
            }
            std::copy(&m_symbols[0], &m_symbols[m_cnt - 1], &m_symbols[1]);
            m_symbols[0].m_sym = c;
            m_symbols[0].m_frq = 0;
        }
        return 0;
    }

    int decode(bit_model_t* see, rc_decoder_t* coder, std::bitset<256>& exclude) {
        auto cum = 0;
        auto frq = 0;

        if (see->decode(coder) == 0) {
            see->update(0);
            if (m_cnt != 1) {  // no need to decode binary context
                auto recent_frq = (m_symbols[0].m_frq + 6) / 2;  // recency scaling
                auto decode_cum = coder->decode_cum(m_sum + recent_frq);
                auto i = 0;
                while (cum + recent_frq + m_symbols[i].m_frq <= decode_cum) {
                    cum += m_symbols[i].m_frq;
                    i++;
                }
                if (i == 0) {
                    frq = m_symbols[i].m_frq + recent_frq;
                } else {
                    frq = m_symbols[i].m_frq;
                    cum += recent_frq;
                    symbol_counter_t tmp_symbol = m_symbols[i];
                    std::copy(&m_symbols[0], &m_symbols[i], &m_symbols[1]);
                    m_symbols[0] = tmp_symbol;
                }
                coder->decode(cum, frq);
            }
            return m_symbols[0].m_sym;

        } else {  // not found
            see->update(1);
            for (auto i = 0; i < m_cnt; i++) {
                exclude[m_symbols[i].m_sym] = 1;  // exclude o4
            }
            if (m_cnt == m_symbols.size()) {
                m_sum -= m_symbols[m_cnt - 1].m_frq;
            } else {
                m_cnt += 1;
            }
            std::copy(&m_symbols[0], &m_symbols[m_cnt - 1], &m_symbols[1]);
            m_symbols[0].m_frq = 0;
        }
        return -1;
    }

    void update(dense_model_t* lower_o2, int c) {
        if (m_symbols[0].m_frq == 0) {  // calculate init frequency
            auto o2c = symbol_counter_t();
            for (auto i = 0; i < lower_o2->m_cnt; i++) {
                if (lower_o2->m_symbols[i].m_sym == c) {
                    o2c = lower_o2->m_symbols[i];
                    break;
                }
            }
            m_symbols[0].m_frq = 2 + (o2c.m_frq * 16 > lower_o2->m_sum);
            m_symbols[0].m_sym = c;
            m_sum += m_symbols[0].m_frq;
        } else {
            auto inc = 1 + (m_symbols[0].m_frq <= 3) + (m_symbols[0].m_frq <= 220);
            m_symbols[0].m_sym = c;
            m_symbols[0].m_frq += inc;
            m_sum += inc;
        }

        if (m_symbols[0].m_frq > 250) {  // rescale
            auto n = 0;
            m_cnt = 0;
            m_sum = 0;
            for (auto i = 0; i + n < m_symbols.size(); i++) {
                if ((m_symbols[i].m_frq = m_symbols[i + n].m_frq / 2) > 0) {
                    m_symbols[i].m_sym = m_symbols[i + n].m_sym;
                    m_cnt += 1;
                    m_sum += m_symbols[i].m_frq;
                } else {
                    n++;
                    i--;
                }
            }
            std::fill(m_symbols.begin() + m_cnt, m_symbols.end(), symbol_counter_t());
        }
        return;
    }
} __attribute__((__aligned__(128)));

// main ppm-model type
struct ppm_model_t {
    std::array<bit_model_t, PPM_SEE_SIZE> m_see;
    std::array<sparse_model_t*, PPM_O4_BUCKET_SIZE> m_o4_buckets;
    std::array<dense_model_t, 65536> m_o2;
    std::array<dense_model_t, 256> m_o1;
    std::array<dense_model_t, 1> m_o0;
    uint32_t m_o4_count;
    uint64_t m_context;
    uint8_t m_see_ch_context;
    uint8_t m_see_last_esc;

    ppm_model_t():
        m_o4_count(0),
        m_context(0),
        m_see_ch_context(0),
        m_see_last_esc(0) {

        for (auto i = 0; i < PPM_SEE_SIZE; i++) {
            m_see[i].m_c[0] = 20;
            m_see[i].m_c[1] = 10;
        }
    }

    bit_model_t* current_see(sparse_model_t* o4) {
        auto log2i = [](uint32_t x) {
            return (31 - __builtin_clz((x << 1) | 0x01));
        };

        if (o4->m_cnt == 0) {
            static bit_model_t see_01 = {{0, 1}};
            return &see_01;  // no symbols under current context -- always escape
        }
        auto curcnt = o4->m_cnt;
        auto lowsum = current_o2()->m_sum;
        auto lowcnt = current_o2()->m_cnt;
        auto context = 0
            | ((m_context >>  6) & 0x03) << 0
            | ((m_context >> 14) & 0x03) << 2
            | ((m_context >> 22) & 0x03) << 4
            | m_see_last_esc << 6;

        if (curcnt == 1) {
            // QUANTIZE(binary) = (sum[3] | lowcnt[2] | lowsum[1] | bin_symbol[3] | last_esc[1] | previous symbols[6])
            context |= 0
                | (o4->m_symbols[0].m_sym >> 5) << 7
                | (lowsum >= 5) << 10
                | std::min(log2i(curcnt / 2), 3) << 11
                | std::min(log2i(o4->m_sum / 3), 7) << 13
                | 1 << 16;
            return &m_see[context];
        } else {
            // QUANTIZE = (sum[3] | curcnt[2] | lowsum[1] | (lowcnt - curcnt)[3] | last_esc[1] | previous symbols[6])
            context |= 0
                | std::min(log2i(std::max(lowcnt - curcnt, 0) / 2), 3) << 7
                | (lowsum >= 5) << 10
                | std::min(log2i(curcnt / 2), 3) << 11
                | std::min(log2i(o4->m_sum / 8), 7) << 13
                | 0 << 16;
            return &m_see[context];
        }
        return nullptr;
    }

    sparse_model_t* current_o4() {
        if (m_o4_count >= PPM_O4_BUCKET_SIZE * 5) {  // too many o4-context/symbol nodes?
            for (auto bucket: m_o4_buckets) {
                auto it0 = bucket;
                auto it1 = bucket ? bucket->m_next : NULL;
                while (it1) {  // clear nodes: non most recent nodes with visited=1
                    if ((it1->m_visited /= 2) == 0) {
                        it0->m_next = it1->m_next;
                        delete it1;
                        m_o4_count -= 1;
                        it1 = it0->m_next;
                        continue;
                    }
                    it0 = it1;
                    it1 = it1->m_next;
                }
            }
        }

        auto compacted_context = 0 | (m_context & 0xc0ffffffffff);

        auto& bucket = m_o4_buckets[((compacted_context >> 16) * 13131 + compacted_context) % PPM_O4_BUCKET_SIZE];
        auto it0 = bucket;
        auto it1 = bucket;
        while (it1 != nullptr) {
            if (it1->m_context == compacted_context) {  // found -- bring to front
                if (it1 != bucket) {
                    it0->m_next = it1->m_next;
                    it1->m_next = bucket;
                    bucket = it1;
                }
                it1->m_visited += (it1->m_visited < 255);
                return it1;
            }
            it0 = it1;
            it1 = it1->m_next;
        }
        auto new_node = new sparse_model_t();  // not found -- create a new one
        new_node->m_context = compacted_context;
        new_node->m_visited = 1;
        new_node->m_next = bucket;
        bucket = new_node;
        m_o4_count++;
        return new_node;
    }
    dense_model_t* current_o2() { return &m_o2[m_context & 0xffff]; }
    dense_model_t* current_o1() { return &m_o1[m_context & 0x00ff]; }
    dense_model_t* current_o0() { return &m_o0[0]; }

    void encode(rc_encoder_t* coder, int c) {
        auto o4 = current_o4();
        auto o2 = current_o2();
        auto o1 = current_o1();
        auto o0 = current_o0();
        auto order = 0;
        auto exclude = std::bitset<256>();

        while (-1) {
            order = 4; if (o4->encode(current_see(o4), coder, c, exclude)) break;
            order = 2; if (o2->encode(coder, exclude, c)) break;
            order = 1; if (o1->encode(coder, exclude, c)) break;
            order = 0; if (o0->encode(coder, exclude, c)) break;

            // decode with o(-1)
            auto cum = 0;
            for (auto i = 0; i < c; i++) {
                cum += !exclude[i];
            }
            coder->encode(cum, 1, 256 - exclude.count());
            break;
        }
        switch (order) {  // fall-through switch
            case 0: o0->update(c);
            case 1: o1->update(c);
            case 2: o2->update(c);
            case 4: o4->update(o2, c);
        }
        m_see_last_esc = (order == 4);
    }

    // main ppm-decode method
    int decode(rc_decoder_t* coder) {
        auto o4 = current_o4();
        auto o2 = current_o2();
        auto o1 = current_o1();
        auto o0 = current_o0();
        auto order = 0;
        auto c = 0;
        auto exclude = std::bitset<256>();

        while (-1) {
            order = 4; if ((c = o4->decode(current_see(o4), coder, exclude)) != -1) break;
            order = 2; if ((c = o2->decode(coder, exclude)) != -1) break;
            order = 1; if ((c = o1->decode(coder, exclude)) != -1) break;
            order = 0; if ((c = o0->decode(coder, exclude)) != -1) break;

            // decode with o(-1)
            auto decode_cum = coder->decode_cum(256 - exclude.count());
            auto cum = 0;
            for (c = 0; cum + !exclude[c] <= decode_cum; c++) {
                cum += !exclude[c];
            }
            coder->decode(cum, 1);
            break;
        }
        switch (order) {  // fall-through switch
            case 0: o0->update(c);
            case 1: o1->update(c);
            case 2: o2->update(c);
            case 4: o4->update(o2, c);
        }
        m_see_last_esc = (order == 4);
        return c;
    }

    void update_context(int c) {
        m_context = m_context << 8 | c;
    }
};

/*******************************************************************************
 * Matcher
 ******************************************************************************/
struct matcher_t {
    static const auto match_min = 12;
    static const auto match_max = 255;
    std::array<uint64_t, 1048576> m_lzp;  // lzp = pos[32] + checksum[16] + prefetch[16]

    matcher_t() {
        m_lzp.fill(0);
    }

    static uint32_t hash2(unsigned char* p) {
        return uint32_t(p[1] * 1919191 + p[0]) % 1048576;
    }
    static uint32_t hash5(unsigned char* p) {
        return uint32_t(p[0] * 1717171 + p[1] * 17171 + p[2] * 171 + p[3]) % 1048576;
    }
    static uint32_t hash8(unsigned char* p) {
        return uint32_t(
                p[0] * 13131313 + p[1] * 1313131 + p[2] * 131313 + p[3] * 13131 +
                p[4] * 1313     + p[5] * 131     + p[6] * 13     + p[7] * 1) % 1048576;
    }

    uint64_t getlzp(unsigned char* data, uint32_t pos) {
        if (pos >= 8) {
            auto lzp8 = m_lzp[hash8(data + pos - 8)];
            auto lzp5 = m_lzp[hash5(data + pos - 5)];
            auto lzp2 = m_lzp[hash2(data + pos - 2)];
            if ((lzp8 >> 32 & 0xffff) == *(uint16_t*)(data + pos - 2) && (lzp8 & 0xffffffff) != 0) return lzp8;
            if ((lzp5 >> 32 & 0xffff) == *(uint16_t*)(data + pos - 2) && (lzp5 & 0xffffffff) != 0) return lzp5;
            if ((lzp2 >> 32 & 0xffff) == *(uint16_t*)(data + pos - 2) && (lzp2 & 0xffffffff) != 0) return lzp2;
        }
        return 0;
    }

    uint32_t getpos(unsigned char* data, uint32_t pos) {
        return getlzp(data, pos) & 0xffffffff;
    }

    uint32_t lookup(unsigned char* data, uint32_t data_size, uint32_t pos, int do_lazy_match = 1, int maxlen = match_max) {
        auto match_lzp = getlzp(data, pos);
        if ((match_lzp >> 48 & 0xffff) != *(uint16_t*)(data + pos + match_min - 2)) {
            return 1;
        }
        auto match_pos = match_lzp & 0xffffffff;
        auto match_len = 0;
        if (match_pos > 0) {
            while (match_pos + match_len < data_size
                    && match_len < maxlen
                    && data[match_pos + match_len] == data[pos + match_len]) {
                match_len++;
            }
        }
        if (do_lazy_match) {
            auto next_match_len = lookup(data, data_size, pos + 1, 0, match_len + 2);
            if (match_len + 1 < next_match_len) {
                return 1;
            }
        }
        return (match_len >= match_min) ? match_len : 1;
    }

    void update(unsigned char* data, uint32_t pos) {
        if (pos >= 8) {  // avoid overflow
            (m_lzp[hash8(data + pos - 8)] =
             m_lzp[hash5(data + pos - 5)] =
             m_lzp[hash2(data + pos - 2)] = (0
                 | (uint64_t) pos
                 | (uint64_t) *(uint16_t*) (data + pos - 2) << 32
                 | (uint64_t) *(uint16_t*) (data + pos + match_min - 2) << 48));
        }
    }
};

/*******************************************************************************
 * Codec
 ******************************************************************************/
static const auto BLOCK_SIZE = 16777216;
static const auto MATCH_LENS_SIZE = 64000;

void zmolly_encode(std::istream& orig, std::ostream& comp) {
    auto ppm = std::make_unique<ppm_model_t>();
    auto orig_data = std::make_unique<unsigned char[]>(BLOCK_SIZE);

    while (orig.peek() != EOF) {
        orig.read((char*) &orig_data[0], BLOCK_SIZE);
        auto orig_size = orig.gcount();

        // find escape char
        auto counts = std::array<int, 256>();
        auto escape = 0;
        for (auto i = 0; i < orig_size; i++) {
            counts[orig_data[i]]++;
        }
        for (auto i = 0; i < 256; i++) {
            escape = counts[escape] < counts[i] ? escape : i;
        }

        auto comp_start_pos = comp.tellp();
        auto matcher = std::make_unique<matcher_t>();
        comp.put(escape);

        auto coder = rc_encoder_t(comp);
        auto orig_pos = size_t(0);

        auto match_idx = 0;
        auto match_pos = 0;
        auto thread = std::thread();
        auto match_lens1 = std::array<int, MATCH_LENS_SIZE>();
        auto match_lens2 = std::array<int, MATCH_LENS_SIZE>();
        auto match_lens_current = &match_lens1;
        auto func_matching_thread = [&](auto match_lens) {
            auto match_idx = 0;
            while (std::streampos(match_pos) < orig_size && match_idx < MATCH_LENS_SIZE) {
                auto match_len = matcher->lookup(&orig_data[0], orig_size, match_pos);
                for (auto i = 0; i < match_len; i++) {
                    matcher->update(&orig_data[0], match_pos + i);
                }
                match_pos += match_len;
                match_lens[match_idx++] = match_len;
            }
        };

        // start thread (matching first block)
        thread = std::thread(func_matching_thread, &match_lens1[0]); thread.join();
        thread = std::thread(func_matching_thread, &match_lens2[0]);

        while (orig_pos < orig_size) {
            // find match in separated thread
            if (match_idx >= MATCH_LENS_SIZE) {  // start the next matching thread
                thread.join();
                thread = std::thread(func_matching_thread, &match_lens_current->operator[](0));
                match_lens_current = (*match_lens_current == match_lens1) ? &match_lens2 : &match_lens1;
                match_idx = 0;
            }
            auto match_len = match_lens_current->operator[](match_idx++);

            if (match_len > 1) {  // encode a match
                ppm->encode(&coder, escape);
                ppm->update_context(escape);
                ppm->encode(&coder, match_len);
                ppm->update_context(match_len);
                for (auto i = 0; i < match_len; i++) {
                    ppm->update_context(orig_data[orig_pos++]);
                }

            } else {  // encode a literal
                ppm->encode(&coder, orig_data[orig_pos]);
                ppm->update_context(orig_data[orig_pos]);
                if (orig_data[orig_pos] == escape) {
                    ppm->encode(&coder, 0);
                    ppm->update_context(0);
                }
                orig_pos++;
            }
        }
        thread.join();
        ppm->encode(&coder, escape);  // write end of block code
        ppm->update_context(escape);
        ppm->encode(&coder, orig.peek() != EOF ? 1 : 2);  // 1: end of block, 2: end of input
        coder.flush();
        fprintf(stderr, "encode-block: %zu => %zu\n", orig_pos, size_t(comp.tellp() - comp_start_pos));
    }
}

void zmolly_decode(std::istream& comp, std::ostream& orig) {
    auto ppm = std::make_unique<ppm_model_t>();
    auto end_of_input = false;
    auto orig_data = std::make_unique<unsigned char[]>(BLOCK_SIZE + 1024);

    while (!end_of_input) {
        auto end_of_block = false;
        auto comp_start_pos = comp.tellg();
        auto matcher = std::make_unique<matcher_t>();
        auto escape = comp.get();
        auto coder = rc_decoder_t(comp);
        auto orig_pos = size_t(0);

        while (!end_of_block) {
            auto c = ppm->decode(&coder);
            ppm->update_context(c);
            if (c != escape) {  // literal
                orig_data[orig_pos] = c;
                matcher->update(&orig_data[0], orig_pos);
                orig_pos++;
            } else {
                auto match_len = ppm->decode(&coder);
                if (match_len >= matcher_t::match_min && match_len <= matcher_t::match_max) {  // match
                    auto match_pos = matcher->getpos(&orig_data[0], orig_pos);
                    for (auto i = 0; i < match_len; i++) {  // update context
                        orig_data[orig_pos] = orig_data[match_pos];
                        ppm->update_context(orig_data[orig_pos]);
                        matcher->update(&orig_data[0], orig_pos);
                        orig_pos++;
                        match_pos++;
                    }
                } else if (match_len == 0) {  // escape literal
                    orig_data[orig_pos] = escape;
                    ppm->update_context(orig_data[orig_pos]);
                    matcher->update(&orig_data[0], orig_pos);
                    orig_pos++;
                } else if (match_len == 1) {  // end of block
                    end_of_block = true;
                } else if (match_len == 2) {  // end of block
                    end_of_block = true;
                    end_of_input = true;
                } else {
                    throw std::runtime_error("invalid input data");
                }
            }
            if (orig_pos > BLOCK_SIZE) {
                throw std::runtime_error("invalid input data");
            }
        }
        orig.write((char*) &orig_data[0], orig_pos);
        fprintf(stderr, "decode-block: %zu <= %zu\n", orig_pos, size_t(comp.tellg() - comp_start_pos));
    }
}

/*******************************************************************************
 * Main
 ******************************************************************************/
int main(int argc, char** argv) {
    fprintf(stderr,
            "zmolly:\n"
            "  simple LZP/PPM data compressor.\n"
            "  author: Zhang Li <richselian@gmail.com>\n"
            "usage:\n"
            "  encode: zmolly e inputFile outputFile\n"
            "  decode: zmolly d inputFile outputFile\n");

    // check args
    if (argc != 4) {
        throw std::runtime_error(std::string() + "invalid number of arguments");
    }
    if (std::string() + argv[1] != "e" && std::string() + argv[1] != std::string("d")) {
        throw std::runtime_error(std::string() + "error: invalid mode: " + argv[1]);
    }

    // open input file
    std::ifstream fin(argv[2], std::ios::in | std::ios::binary);
    fin.exceptions(std::ios_base::failbit);
    if (!fin.is_open()) {
        throw std::runtime_error(std::string() + "cannot open input file: " + argv[2]);
    }

    // open output file
    std::ofstream fout(argv[3], std::ios::out | std::ios::binary);
    fin.exceptions(std::ios_base::failbit);
    if (!fout.is_open()) {
        throw std::runtime_error(std::string() + "cannot open output file: " + argv[3]);
    }

    // encode/decode
    if (std::string() + argv[1] == "e") zmolly_encode(fin, fout);
    if (std::string() + argv[1] == "d") zmolly_decode(fin, fout);
    return 0;
}
