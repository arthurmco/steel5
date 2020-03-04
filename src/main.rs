use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};

/*
 * Instruction types
 *
 * We have 4 types: R, I, S and U, and 2 additional types: B and J
 * opcode will always have 7 bits, funct3 has 3 bits, rd, rs1 and rs2 will
 * have 5 bits, funct7 has 6 bits, immediate varies
 *
 * R-type instructions are used with register arithmetic instructions
 * I-type ones are used with immediate arithmetic instructions
 * S-type are used with stores
 * B-type are used with branch instructions
 * J-type are used for jumps
 * U-type are used for upper immediate instructions, instructions that
 * need to access a bigger memory area.
 */
#[derive(Debug)]
pub enum InstructionType {
    RInst {
        rd: u32,
        funct3: u32,
        rs1: u32,
        rs2: u32,
        funct7: u32,
    },
    IInst {
        rd: u32,
        funct3: u32,
        rs1: u32,
        imm: i32,
    },
    SInst {
        rd: u32,
        funct3: u32,
        rs1: u32,
        rs2: u32,
        imm: i32,
    },

    // B-type has the same fields as S-tyle, but the immediate
    // is spread in memory like this:
    // | imm[12] | imm[10:5] | rs2 | rs1 | funct3 | imm[4:1] | imm[11] | opcode
    BInst {
        funct3: u32,
        rs1: u32,
        rs2: u32,
        imm: i32,
    },

    // Watch out that we only save the top 20 bits of the immediate
    // in the U-instruction
    UInst {
        rd: u32,
        imm: i32,
    },

    // J-type has the same fields as I type, but the immediate is
    // spread like this, in memory
    // | imm[20] | imm[10:1] | imm[11] | imm[19:12] | rd | opcode |
    JInst {
        rd: u32,
        imm: u32,
    },
}

#[derive(Debug, Copy, Clone, ToPrimitive, FromPrimitive)]
pub enum OpcodeImmFunctions {
    Value = 0b0010011,

    // Adds imm to value in register rs1
    ADDI = 0b000,

    // Puts 1 in register rd if value in rs1 is less than the immediate
    // else puts 0
    SLTI = 0b010,

    // Same as SLTI, but comparison is unsigned
    SLTIU = 0b011,

    // rd = rs1 AND imm
    ANDI = 0b111,

    // rd = rs1 OR imm
    ORI = 0b110,

    // rd = rs1 XOR imm
    XORI = 0b100,

    // Shifts rs1 by the amount on the lower 5 bits of imm
    // SLLI = shift left,
    SLLI = 0b001,

    // SRLI, shift right,
    // SRAI = arithmetical right shift (preserves sign bit)
    // The higher 7 bits of imm determine if we have a SRLI or SRAI
    // SRLI is 0, SRAI is 0b100000
    SRLI_SRAI = 0b101,
}

#[derive(Debug, Copy, Clone, ToPrimitive, FromPrimitive)]
pub enum OpcodeRegFunctions {
    Value = 0b0110011,

    // Adds rs1 to rs2, puts result into rd
    // Subtracts rs2 from rs1
    // The higher bits of rs2 determine if we have a ADD or SUB
    // ADD is 0, SUB is 0b100000
    ADD_SUB = 0b000,

    // Same as SLTI, but compares rs1 with rs2 instead of rs1 and imm
    SLT = 0b010,
    SLTU = 0b011,

    // rd = rs1 AND rs2
    AND = 0b111,

    // rd = rs1 OR rs2
    OR = 0b110,

    // rd = rs1 XOR rs2
    XOR = 0b100,

    // SLL = same as SLLI, but shifts rs1 by lower 5 bits of rs2
    SLL = 0b001,

    // Same as SRLI, but shifts right by lower 5 bits of rs2
    // Same as SRAI but... you understood.
    // The higher bits of rs2 determine if we have a SRL or SRA
    // SRL is 0, SRA is 0b100000
    SRL_SRA = 0b101,
}

// All branch instructions, when branching, will get the
// 12-bit offset from the immediate, sign extend it, add
// to pc and jump to it.
#[derive(Debug, Copy, Clone, ToPrimitive, FromPrimitive)]
pub enum BranchFunctions {
    Value = 0b1100011,

    // Branch if rs1 and rs2 are equal.
    BEQ = 0,

    // Branch if rs1 and rs2 are different
    BNE = 0b001,

    // Branch if rs1 < rs2, signed compare
    BLT = 0b100,

    // Branch if rs1 < rs2, unsigned compare
    BLTU = 0b110,

    // Branch if rs1 >= rs2, signed cmp
    BGE = 0b101,

    // Branch if rs1 >= rs2, unsigned cmp
    BGEU = 0b111,
}

#[derive(Debug, Copy, Clone, ToPrimitive)]
pub enum FreeOpcodes {
    /**** I-type ****/
    // Jumps to the address specified by rs1 + imm, imm being
    // signed. The address least significant bit is set to
    // zero before jumping to it. rd becomes the address
    // of the instruction following the jump.
    // This generates a instruction address misaligned
    // exception of the instruction is not aligned to a
    // 4-byte boundary
    JALR = 0b1100111,

    // Loads a value from memory address (rs1 + offset) to rd
    // funct3 stores the load width (0=byte, 1=word, 2=dword)
    // The MSB means that the load is unsigned.
    // The byte, halfword and word load mnemonics are LB, LH and LW, the unsigned
    // variants are LBU, LHU and LWU
    LOAD = 0b0000011,

    /**** S-type ****/
    // Stores the value in rs2 into memory address (rs1 + offset)
    // funct3 stores the store width (0=byte, 1=halfword, 2=word)
    // The byte, halfword and word store mnemonics are SB, SH and SW
    STORE = 0b0100011,

    /**** U-type ****/
    // Loads the immediate value in the top 20 bits of rd,
    // and fills the lower 12 bits with zeros
    // LUI means "load upper immediate"
    LUI = 0b0110111,

    // Forms a 32 bit offset from the 20-bit immediate,
    // fills the lower 12 bits with zeroes, adds to PC
    // and puts the result in rd.
    // AUIPC means "add upper immediate to PC"
    AUIPC = 0b0010111,

    /**** J-type ****/
    // Jumps to the offset specified by the immediate
    // (in reality, jumps to pc + offset). The offset is sign
    // extended. Stores the address of the following instruction
    // (pc+4) into rd. Usually rd is x1.
    // Unconditional jumps (j <offset>) are encoded as "jal x0, <offset>"
    JAL = 0b1101111,
}

/*
 * The opcodes.
 *
 * Sometimes the opcode will have some enum inside it, the enum values are the values of funct3 for
 * that specific opcode.
 *
 * The RISCV has a NOP instruction, who is basically
 * "addi r0, r0, 0"
 */
#[derive(Debug)]
pub enum Opcode {
    /**** R-type ****/
    OP(OpcodeRegFunctions),

    /**** I-type ****/
    OP_IMM(OpcodeImmFunctions),

    /**** B-type ****/
    BRANCH(BranchFunctions),

    FREE_OPS(FreeOpcodes),
}

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
    opcode: Opcode,
    data: InstructionType,
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
            Opcode::OP(ins) => Instruction::parse_rtype(inst),
            Opcode::OP_IMM(ins) => Instruction::parse_itype(inst),
            Opcode::BRANCH(ins) => Instruction::parse_btype(inst),
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
            _ => panic!("Unknown instruction format"),
        };

        Instruction {
            opcode,
            data: instruction_type,
        }
    }
}

/*
 * Represents a memory area
 *
 * A memory area has a starting address, an ending address and
 * a type, that identifies if this is real memory or some
 * hardware register mapped to memory
 *
 * We use a boxed slice so we do not have to worry much about
 * if the vector is not bigger or smaller than we wanted.
 */
#[derive(Debug)]
pub struct MemoryArea {
    tag: String,
    start: u32,
    end: u32,
    data: Box<[u8]>,
}

impl MemoryArea {
    pub fn new(tag: String, start: u32, end: u32) -> MemoryArea {
        let size = end - start;

        assert_eq!(size > 0, true);
        assert_eq!(size % 4, 0);

        let v: Vec<u8> = vec![0 as u8; size as usize];

        MemoryArea {
            tag,
            start,
            end,
            data: v.into_boxed_slice(),
        }
    }

    fn write8(&mut self, offset: u32, val: u8) {
        self.data[offset as usize] = val
    }

    fn read8(&self, offset: u32) -> u8 {
        self.data[offset as usize]
    }
}

/*
 * Represents the machine's memory address space
 */
#[derive(Debug)]
pub struct Memory {
    areas: Vec<MemoryArea>,
}

impl Memory {
    pub fn new() -> Memory {
        return Memory {
            areas: vec![MemoryArea::new(String::from("main-memory"), 0, 128)],
        };
    }

    ///  Write a byte to an address
    pub fn write8(&mut self, addr: u32, val: u8) {
        match self.find_memory_area_mut(addr) {
            Some(area) => {
                let offset = addr - area.start;
                area.write8(offset, val)
            }
            None => panic!("Cannot find suitable memory area for address {:x}", addr),
        }
    }

    /// Read a byte from an address
    pub fn read8(&self, addr: u32) -> u8 {
        match self.find_memory_area(addr) {
            Some(area) => {
                let offset = addr - area.start;
                area.read8(offset)
            }
            None => panic!("Cannot find suitable memory area for address {:x}", addr),
        }
    }

    /// Read 4 bytes, 32 bits, from an address
    /// Good for reading instructions, since they will usually have 4 bytes
    pub fn read32(&self, addr: u32) -> u32 {
        match self.find_memory_area(addr) {
            Some(area) => {
                let offset = addr - area.start;

                let bytes: Vec<u32> = vec![0, 1, 2, 3]
                    .iter()
                    .map(|i| area.read8(offset + i) as u32)
                    .collect();
                let ret = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | bytes[3] << 24;
                ret
            }
            None => panic!("Cannot find suitable memory area for address {:x}", addr),
        }
    }

    fn find_memory_area_mut(&mut self, addr: u32) -> Option<&mut MemoryArea> {
        self.areas
            .iter_mut()
            .find(|a| addr >= a.start && addr < a.end)
    }

    fn find_memory_area(&self, addr: u32) -> Option<&MemoryArea> {
        self.areas.iter().find(|a| addr >= a.start && addr < a.end)
    }
}

/*
 * Represents a CPU state
 *
 * This includes the registers (x0 to x31) and the program counter
 * Reads on x0 always returns 0, writes have no effect. The other registers are general
 * purpose registers
 *
 * TODO: add floating point registers
 */
#[derive(Debug)]
pub struct CPUState {
    x: [u32; 32],
    pc: u32,
}

impl CPUState {
    pub fn new(pc: u32) -> CPUState {
        return CPUState { x: [0; 32], pc };
    }
}

fn main() {
    println!("Hello, world!");

    // Machine code examples extracted from here
    //  - https://passlab.github.io/CSCE513/notes/lecture04_RISCV_ISA.pdf
    //  - https://inst.eecs.berkeley.edu/~cs61c/fa17/disc/4/Disc4Sol.pdf
    println!("{:?}", Instruction::new(0xFFF28413)); //addi s0 t0 -1
    println!("{:?}", Instruction::new(0x007302b3)); //add x5, x6, x7
    println!("{:?}", Instruction::new(0x00650333)); //add x6, x10, x6
    println!("{:?}", Instruction::new(0x00512a03)); //lw s4 5(sp)
    println!("{:?}", Instruction::new(0x60000113)); //addi sp, 0, 1536
    println!("{:?}", Instruction::new(0x00000013)); //addi x0, x0, 0 aka NOP
    println!("{:?}", Instruction::new(0x0400026f)); //jal x4, #64
    println!("{:?}", Instruction::new(0x0400026f)); //jal x4, #64
    println!("{:?}", Instruction::new(0x02419063)); //bne r3, r4, #32

    let cpu = CPUState::new(0x300000);
    let mut memory = Memory::new();

    memory.write8(0, 1);
    memory.write8(1, 2);
    memory.write8(2, 3);
    memory.write8(3, 4);
    memory.write8(4, 0x20);
    memory.write8(5, 0x21);
    memory.write8(6, 0x22);
    memory.write8(7, 0x23);
    memory.write8(8, 0x24);

    println!("{:?}", cpu);
    println!("{:?}", memory);
    println!("{:x}", memory.read32(0)); // needs to be 0x04030201
    println!("{:x}", memory.read32(4)); // needs to be 0x23222120
    println!("{:x}", memory.read8(8)); // needs to be 0x24
}
