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

    /// Write 2 bytes, 16 bits, to an address
    pub fn write16(&mut self, addr: u32, data: u16) {
        match self.find_memory_area_mut(addr) {
            Some(area) => {
                let offset = addr - area.start;

                let bytes: [u16; 2] = [data & 0xff, (data >> 8) & 0xff];

                self.write8(addr + 0, bytes[0] as u8);
                self.write8(addr + 1, bytes[1] as u8);
            }
            None => panic!("Cannot find suitable memory area for address {:x}", addr),
        }
    }

    /// Read 2 bytes, 16 bits, from an address
    pub fn read16(&self, addr: u32) -> u16 {
        match self.find_memory_area(addr) {
            Some(area) => {
                let offset = addr - area.start;

                let bytes: Vec<u16> = vec![0, 1]
                    .iter()
                    .map(|i| area.read8(offset + i) as u16)
                    .collect();
                let ret = bytes[0] | (bytes[1] << 8);
                ret
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

    /// Write 4 bytes, 32 bits, to an address
    pub fn write32(&mut self, addr: u32, data: u32) {
        match self.find_memory_area_mut(addr) {
            Some(area) => {
                let offset = addr - area.start;

                let bytes: [u32; 4] = [
                    data & 0xff,
                    (data >> 8) & 0xff,
                    (data >> 16) & 0xff,
                    (data >> 24) & 0xff,
                ];

                for i in [0, 1, 2, 3].iter() {
                    self.write8(addr + i, bytes[*i as usize] as u8);
                }
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
