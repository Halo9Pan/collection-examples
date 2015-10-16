/**
 * Created on Nov 21, 2011
 */
package info.halo9pan.examples.prospring3.ch14.domain;

/**
 * @author Clarence
 *
 */
public enum CustomerType {

	INDIVIDUAL("I"), CORPORATE("C");

	private String code;

	private CustomerType(String code) {
		this.code = code;
	}

	public String toString() {
		return this.code;
	}

}
