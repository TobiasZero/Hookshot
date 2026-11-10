// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/bit_field.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {
enum class MaxShift : u64 {
    U32,
    Undefined,
    U64,
    S64,
};

IR::U32 FunnelShift32(IR::IREmitter& ir, const IR::U32& lo, const IR::U32& hi,
                      const IR::U32& shift, bool right_shift, bool is_signed) {
    const IR::U32 c0{ir.Imm32(0)};
    const IR::U32 c32{ir.Imm32(32)};
    const IR::U32 inv_shift{ir.ISub(shift, c32)};
    const IR::U1 is_long{ir.IGreaterThanEqual(inv_shift, c0, true)};
    const IR::U1 is_zero{ir.IEqual(shift, c0)};
    const IR::U32 shift_comp{ir.ISub(c32, shift)};
    if (!right_shift) {
        const IR::U32 shifted_hi{ir.ShiftLeftLogical(hi, shift)};
        const IR::U32 long_ret_hi{ir.ShiftLeftLogical(lo, inv_shift)};
        const IR::U32 lo_extract{ir.BitFieldExtract(lo, shift_comp, shift, false)};
        const IR::U32 short_ret_hi{ir.BitwiseOr(shifted_hi, lo_extract)};
        const IR::U32 non_zero_hi{ir.Select(is_long, long_ret_hi, short_ret_hi)};
        return IR::U32{ir.Select(is_zero, hi, non_zero_hi)};
    }
    const IR::U32 shifted_lo{ir.ShiftRightLogical(lo, shift)};
    const IR::U32 long_ret_lo{is_signed ? ir.ShiftRightArithmetic(hi, inv_shift)
                                        : ir.ShiftRightLogical(hi, inv_shift)};
    const IR::U32 short_hi_extract{ir.BitFieldExtract(hi, c0, shift, false)};
    const IR::U32 short_ret_lo{ir.BitFieldInsert(shifted_lo, short_hi_extract, shift_comp, shift)};
    const IR::U32 non_zero_lo{ir.Select(is_long, long_ret_lo, short_ret_lo)};
    return IR::U32{ir.Select(is_zero, lo, non_zero_lo)};
}

void SHF(TranslatorVisitor& v, u64 insn, const IR::U32& shift, const IR::U32& high_bits,
         bool right_shift) {
    union {
        u64 insn;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<8, 8, IR::Reg> lo_bits_reg;
        BitField<37, 2, MaxShift> max_shift;
        BitField<47, 1, u64> cc;
        BitField<48, 2, u64> x_mode;
        BitField<50, 1, u64> wrap;
    } const shf{insn};

    if (shf.cc != 0) {
        throw NotImplementedException("SHF CC");
    }
    if (shf.x_mode != 0) {
        throw NotImplementedException("SHF X Mode");
    }
    if (shf.max_shift == MaxShift::Undefined) {
        throw NotImplementedException("SHF Use of undefined MaxShift value");
    }
    const IR::U32 low_bits{v.X(shf.lo_bits_reg)};
    const IR::U32 max_shift{shf.max_shift == MaxShift::U32 ? v.ir.Imm32(32) : v.ir.Imm32(63)};
    const IR::U32 safe_shift{shf.wrap != 0
                                 ? v.ir.BitwiseAnd(shift, v.ir.ISub(max_shift, v.ir.Imm32(1)))
                                 : v.ir.UMin(shift, max_shift)};

    const bool is_signed{shf.max_shift == MaxShift::S64};
    const IR::U32 result{FunnelShift32(v.ir, low_bits, high_bits, safe_shift, right_shift, is_signed)};
    v.X(shf.dest_reg, result);
}
} // Anonymous namespace

void TranslatorVisitor::SHF_l_reg(u64 insn) {
    SHF(*this, insn, GetReg20(insn), GetReg39(insn), false);
}

void TranslatorVisitor::SHF_l_imm(u64 insn) {
    SHF(*this, insn, GetImm20(insn), GetReg39(insn), false);
}

void TranslatorVisitor::SHF_r_reg(u64 insn) {
    SHF(*this, insn, GetReg20(insn), GetReg39(insn), true);
}

void TranslatorVisitor::SHF_r_imm(u64 insn) {
    SHF(*this, insn, GetImm20(insn), GetReg39(insn), true);
}

} // namespace Shader::Maxwell
