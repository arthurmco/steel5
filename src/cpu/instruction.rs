use crate::opcodes::*;
use num_traits::{FromPrimitive, ToPrimitive};

/*
 * A RISC-V instruction
 *
 * This includes the opcode plus some additional registers that it uses,
 * or some memory offset, depending on the opcode.
 * We also have fields in the instruction for signaling a future variable-length
 * encoding.
 *
 * All instructions must be aligned on a 32-bit boundary. If the instruction is
 * variable-length, it length must be a multiple of 16-bit.
 *
 * Because of this scheme, all 32-bit RISC-V instructions have the 2-lower order bits
 * set.
 *
 * Also, instructions with the lower 16-bits all zeroed are invalid
 */
#[derive(Debug)]
pub struct Instruction {
    pub opcode: Opcode,
    pub data: InstructionType,
}

impl Instruction {
    /* Parse register and immediate fields for a i-type instruction */
    fn parse_itype(inst: u32) -> InstructionType {
        let rd = (inst >> 7) & 0x1f;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1f;
        let uimm = (inst >> 20) & 0x7ff;

        // Immediates are signed!
        let imm: i32 = if uimm > 1024 {
            (uimm as i32) - 2048
        } else {
            uimm as i32
        };

        InstructionType::IInst {
            rd,
            funct3,
            rs1,
            imm,
        }
    }

    /* Parse register and immediate fields for a S-type instruction */
    fn parse_stype(inst: u32) -> InstructionType {
        let rd = (inst >> 7) & 0x1f;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1f;
        let rs2 = (inst >> 20) & 0x1f;
        let uimm = (inst >> 25) & 0x3f;

        // Immediates are signed!
        let imm: i32 = if uimm > 32 {
            (uimm as i32) - 64
        } else {
            uimm as i32
        };

        InstructionType::SInst {
            rd,
            funct3,
            rs1,
            rs2,
            imm: imm << 5,
        }
    }

    /* Parse register and immediate fields for a R-type instruction */
    fn parse_rtype(inst: u32) -> InstructionType {
        let rd = (inst >> 7) & 0x1f;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1f;
        let rs2 = (inst >> 20) & 0x1f;
        let funct7 = (inst >> 25) & 0x3f;

        InstructionType::RInst {
            rd,
            funct3,
            rs1,
            rs2,
            funct7,
        }
    }

    /* Parse register and immediate fields for a U-type instruction */
    fn parse_utype(inst: u32) -> InstructionType {
        let rd = (inst >> 7) & 0x1f;

        let uimm = (inst >> 12) & 0xfffff;

        // Immediates are signed!
        let imm: i32 = if uimm > 0x80000 {
            (uimm as i32) - 0x100000
        } else {
            uimm as i32
        };

        InstructionType::UInst { rd, imm: imm << 12 }
    }

    /* Parse register and immediate fields for a B-type instruction */
    fn parse_btype(inst: u32) -> InstructionType {
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1f;
        let rs2 = (inst >> 20) & 0x1f;

        let imm11 = (inst >> 7) & 0x1;
        let imm1 = (inst >> 8) & 0xf;
        let imm5 = (inst >> 25) & 0x3f;
        let imm12 = (inst >> 31) & 0x1;

        let uimm = (imm1 << 1) | (imm5 << 5) | (imm11 << 11) | (imm12 << 12);

        // Immediates are signed!
        let imm: i32 = if uimm > 2048 {
            (uimm as i32) - 4096
        } else {
            uimm as i32
        };

        InstructionType::BInst {
            funct3,
            rs1,
            rs2,
            imm,
        }
    }

    /**
     * Decode a instruction from a 4-bit integer
     *
     * This function will have to be adapted for variable length instructions
     */
    pub fn new(inst: u32) -> Instruction {
        let opcode_val = inst & ((1 << 7) - 1);
        let funct3 = (inst >> 12) & 0x7;

        let ops = [
            ToPrimitive::to_u32(&OpcodeRegFunctions::Value).unwrap(),
            ToPrimitive::to_u32(&OpcodeImmFunctions::Value).unwrap(),
            ToPrimitive::to_u32(&BranchFunctions::Value).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::JALR).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::STORE).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::LUI).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::AUIPC).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::JAL).unwrap(),
            ToPrimitive::to_u32(&FreeOpcodes::LOAD).unwrap(),
        ];

        let opcode: Opcode = match opcode_val {
            _ if opcode_val == ops[0] => Opcode::OP(match FromPrimitive::from_u32(funct3) {
                Some(OpcodeRegFunctions::ADD_SUB) => OpcodeRegFunctions::ADD_SUB,
                Some(OpcodeRegFunctions::SLT) => OpcodeRegFunctions::SLT,
                Some(OpcodeRegFunctions::SLTU) => OpcodeRegFunctions::SLTU,
                Some(OpcodeRegFunctions::AND) => OpcodeRegFunctions::AND,
                Some(OpcodeRegFunctions::OR) => OpcodeRegFunctions::OR,
                Some(OpcodeRegFunctions::XOR) => OpcodeRegFunctions::XOR,
                Some(OpcodeRegFunctions::SLL) => OpcodeRegFunctions::SLL,
                Some(OpcodeRegFunctions::SRL_SRA) => OpcodeRegFunctions::SRL_SRA,
                _ => panic!("Unknown funct3 value in OP_IMM"),
            }),
            _ if opcode_val == ops[1] => Opcode::OP_IMM(match FromPrimitive::from_u32(funct3) {
                Some(OpcodeImmFunctions::ADDI) => OpcodeImmFunctions::ADDI,
                Some(OpcodeImmFunctions::SLTI) => OpcodeImmFunctions::SLTI,
                Some(OpcodeImmFunctions::SLTIU) => OpcodeImmFunctions::SLTIU,
                Some(OpcodeImmFunctions::ANDI) => OpcodeImmFunctions::ANDI,
                Some(OpcodeImmFunctions::ORI) => OpcodeImmFunctions::ORI,
                Some(OpcodeImmFunctions::XORI) => OpcodeImmFunctions::XORI,
                Some(OpcodeImmFunctions::SLLI) => OpcodeImmFunctions::SLLI,
                Some(OpcodeImmFunctions::SRLI_SRAI) => OpcodeImmFunctions::SRLI_SRAI,
                _ => panic!("Unknown funct3 value in OP_IMM"),
            }),
            _ if opcode_val == ops[2] => Opcode::BRANCH(match FromPrimitive::from_u32(funct3) {
                Some(BranchFunctions::BEQ) => BranchFunctions::BEQ,
                Some(BranchFunctions::BNE) => BranchFunctions::BNE,
                Some(BranchFunctions::BLT) => BranchFunctions::BLT,
                Some(BranchFunctions::BLTU) => BranchFunctions::BLTU,
                Some(BranchFunctions::BGE) => BranchFunctions::BGE,
                Some(BranchFunctions::BGEU) => BranchFunctions::BGEU,
                _ => panic!("Unknown funct3 value in BRANCH"),
            }),
            _ if opcode_val == ops[3] => Opcode::FREE_OPS(FreeOpcodes::JALR),
            _ if opcode_val == ops[4] => Opcode::FREE_OPS(FreeOpcodes::STORE),
            _ if opcode_val == ops[5] => Opcode::FREE_OPS(FreeOpcodes::LUI),
            _ if opcode_val == ops[6] => Opcode::FREE_OPS(FreeOpcodes::AUIPC),
            _ if opcode_val == ops[7] => Opcode::FREE_OPS(FreeOpcodes::JAL),
            _ if opcode_val == ops[8] => Opcode::FREE_OPS(FreeOpcodes::LOAD),
            _ => panic!("Unrecognized opcode"),
        };

        let instruction_type = match opcode {
            Opcode::OP(_) => Instruction::parse_rtype(inst),
            Opcode::OP_IMM(_) => Instruction::parse_itype(inst),
            Opcode::BRANCH(_) => Instruction::parse_btype(inst),
            Opcode::FREE_OPS(ins) => {
                match ins {
                    FreeOpcodes::JALR => Instruction::parse_itype(inst),
                    FreeOpcodes::STORE => Instruction::parse_stype(inst),
                    FreeOpcodes::LUI => Instruction::parse_utype(inst),
                    FreeOpcodes::AUIPC => Instruction::parse_utype(inst),
                    FreeOpcodes::JAL => {
                        // J-type
                        let rd = (inst >> 7) & 0x1f;

                        let imm12 = (inst >> 12) & ((1 << 7) - 1);
                        let imm11 = (inst >> 20) & 0x1;
                        let imm1 = (inst >> 21) & ((1 << 9) - 1);
                        let imm20 = (inst >> 31) & 0x1;

                        let imm = (imm1 << 1) | (imm11 << 11) | (imm12 << 12) | (imm20 << 20);

                        InstructionType::JInst { rd, imm }
                    }
                    FreeOpcodes::LOAD => Instruction::parse_itype(inst),
                }
            }
        };

        Instruction {
            opcode,
            data: instruction_type,
        }
    }
}
